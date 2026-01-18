import abc
import json
import logging
import pathlib
from collections import defaultdict
from enum import Enum
from dataclasses import dataclass, field
from typing import (
import ray
from ray.rllib.core.learner.reduce_result_dict_fn import _reduce_mean_results
from ray.rllib.core.learner.scaling_config import LearnerGroupScalingConfig
from ray.rllib.core.rl_module.marl_module import (
from ray.rllib.core.rl_module.rl_module import (
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, MultiAgentBatch
from ray.rllib.utils.annotations import (
from ray.rllib.utils.debug import update_global_seed_if_necessary
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.metrics import (
from ray.rllib.utils.minibatch_utils import (
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.serialization import serialize_type
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
@OverrideToImplementCustomLogic
def additional_update(self, *, module_ids_to_update: Sequence[ModuleID]=None, timestep: int, **kwargs) -> Mapping[ModuleID, Any]:
    """Apply additional non-gradient based updates to this Algorithm.

        For example, this could be used to do a polyak averaging update
        of a target network in off policy algorithms like SAC or DQN.

        Example:

        .. testcode::

            from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import (
                PPOTorchRLModule
            )
            from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
            from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import (
                PPOTorchLearner
            )
            from ray.rllib.algorithms.ppo.ppo_learner import (
                LEARNER_RESULTS_CURR_KL_COEFF_KEY
            )
            from ray.rllib.algorithms.ppo.ppo_learner import PPOLearnerHyperparameters
            import gymnasium as gym

            env = gym.make("CartPole-v1")
            hps = PPOLearnerHyperparameters(
                use_kl_loss=True,
                kl_coeff=0.2,
                kl_target=0.01,
                use_critic=True,
                clip_param=0.3,
                vf_clip_param=10.0,
                entropy_coeff=0.01,
                entropy_coeff_schedule = [
                    [0, 0.01],
                    [20000000, 0.0],
                ],
                vf_loss_coeff=0.5,
            )

            # Create a single agent RL module spec.
            module_spec = SingleAgentRLModuleSpec(
                module_class=PPOTorchRLModule,
                observation_space=env.observation_space,
                action_space=env.action_space,
                model_config_dict = {"hidden": [128, 128]},
                catalog_class = PPOCatalog,
            )

            class CustomPPOLearner(PPOTorchLearner):
                def additional_update_for_module(
                    self, *, module_id, hps, timestep, sampled_kl_values
                ):

                    results = super().additional_update_for_module(
                        module_id=module_id,
                        hps=hps,
                        timestep=timestep,
                        sampled_kl_values=sampled_kl_values,
                    )

                    # Try something else than the PPO paper here.
                    sampled_kl = sampled_kl_values[module_id]
                    curr_var = self.curr_kl_coeffs_per_module[module_id]
                    if sampled_kl > 1.2 * self.hps.kl_target:
                        curr_var.data *= 1.2
                    elif sampled_kl < 0.8 * self.hps.kl_target:
                        curr_var.data *= 0.4
                    results.update({LEARNER_RESULTS_CURR_KL_COEFF_KEY: curr_var.item()})


            learner = CustomPPOLearner(
                module_spec=module_spec,
                learner_hyperparameters=hps,
            )

            # Note: the learner should be built before it can be used.
            learner.build()

            # Inside a training loop, we can now call the additional update as we like:
            for i in range(100):
                # sample = ...
                # learner.update(sample)
                if i % 10 == 0:
                    learner.additional_update(
                        timestep=i,
                        sampled_kl_values={"default_policy": 0.5}
                    )

        Args:
            module_ids_to_update: The ids of the modules to update. If None, all
                modules will be updated.
            timestep: The current timestep.
            **kwargs: Keyword arguments to use for the additional update.

        Returns:
            A dictionary of results from the update
        """
    results_all_modules = {}
    module_ids = module_ids_to_update or self.module.keys()
    for module_id in module_ids:
        module_results = self.additional_update_for_module(module_id=module_id, hps=self.hps.get_hps_for_module(module_id), timestep=timestep, **kwargs)
        results_all_modules[module_id] = module_results
    return results_all_modules