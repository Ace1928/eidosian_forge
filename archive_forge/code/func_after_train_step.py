import dataclasses
from typing import Optional, Type
import logging
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.appo.appo_learner import (
from ray.rllib.algorithms.impala.impala import Impala, ImpalaConfig
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics import ALL_MODULES, LEARNER_STATS_KEY
from ray.rllib.utils.typing import (
def after_train_step(self, train_results: ResultDict) -> None:
    """Updates the target network and the KL coefficient for the APPO-loss.

        This method is called from within the `training_step` method after each train
        update.
        The target network update frequency is calculated automatically by the product
        of `num_sgd_iter` setting (usually 1 for APPO) and `minibatch_buffer_size`.

        Args:
            train_results: The results dict collected during the most recent
                training step.
        """
    if self.config._enable_new_api_stack:
        if NUM_TARGET_UPDATES in train_results:
            self._counters[NUM_TARGET_UPDATES] += train_results[NUM_TARGET_UPDATES]
            self._counters[LAST_TARGET_UPDATE_TS] = train_results[LAST_TARGET_UPDATE_TS]
    else:
        last_update = self._counters[LAST_TARGET_UPDATE_TS]
        cur_ts = self._counters[NUM_AGENT_STEPS_SAMPLED if self.config.count_steps_by == 'agent_steps' else NUM_ENV_STEPS_SAMPLED]
        target_update_freq = self.config.num_sgd_iter * self.config.minibatch_buffer_size
        if cur_ts - last_update > target_update_freq:
            self._counters[NUM_TARGET_UPDATES] += 1
            self._counters[LAST_TARGET_UPDATE_TS] = cur_ts
            self.workers.local_worker().foreach_policy_to_train(lambda p, _: p.update_target())
            if self.config.use_kl_loss:

                def update(pi, pi_id):
                    assert LEARNER_STATS_KEY not in train_results, ('{} should be nested under policy id key'.format(LEARNER_STATS_KEY), train_results)
                    if pi_id in train_results:
                        kl = train_results[pi_id][LEARNER_STATS_KEY].get('kl')
                        assert kl is not None, (train_results, pi_id)
                        pi.update_kl(kl)
                    else:
                        logger.warning('No data for {}, not updating kl'.format(pi_id))
                self.workers.local_worker().foreach_policy_to_train(update)