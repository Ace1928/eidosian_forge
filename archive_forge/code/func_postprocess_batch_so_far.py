import collections
import logging
import numpy as np
from typing import List, Any, Dict, Optional, TYPE_CHECKING
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.typing import PolicyID, AgentID
from ray.util.debug import log_once
def postprocess_batch_so_far(self, episode: Optional[Episode]=None) -> None:
    """Apply policy postprocessors to any unprocessed rows.

        This pushes the postprocessed per-agent batches onto the per-policy
        builders, clearing per-agent state.

        Args:
            episode (Optional[Episode]): The Episode object that
                holds this MultiAgentBatchBuilder object.
        """
    pre_batches = {}
    for agent_id, builder in self.agent_builders.items():
        pre_batches[agent_id] = (self.policy_map[self.agent_to_policy[agent_id]], builder.build_and_reset())
    post_batches = {}
    if self.clip_rewards is True:
        for _, (_, pre_batch) in pre_batches.items():
            pre_batch['rewards'] = np.sign(pre_batch['rewards'])
    elif self.clip_rewards:
        for _, (_, pre_batch) in pre_batches.items():
            pre_batch['rewards'] = np.clip(pre_batch['rewards'], a_min=-self.clip_rewards, a_max=self.clip_rewards)
    for agent_id, (_, pre_batch) in pre_batches.items():
        other_batches = pre_batches.copy()
        del other_batches[agent_id]
        policy = self.policy_map[self.agent_to_policy[agent_id]]
        if not pre_batch.is_single_trajectory() or len(set(pre_batch[SampleBatch.EPS_ID])) > 1:
            raise ValueError('Batches sent to postprocessing must only contain steps from a single trajectory.', pre_batch)
        post_batches[agent_id] = pre_batch
        if getattr(policy, 'exploration', None) is not None:
            policy.exploration.postprocess_trajectory(policy, post_batches[agent_id], policy.get_session())
        post_batches[agent_id] = policy.postprocess_trajectory(post_batches[agent_id], other_batches, episode)
    if log_once('after_post'):
        logger.info('Trajectory fragment after postprocess_trajectory():\n\n{}\n'.format(summarize(post_batches)))
    from ray.rllib.evaluation.rollout_worker import get_global_worker
    for agent_id, post_batch in sorted(post_batches.items()):
        self.callbacks.on_postprocess_trajectory(worker=get_global_worker(), episode=episode, agent_id=agent_id, policy_id=self.agent_to_policy[agent_id], policies=self.policy_map, postprocessed_batch=post_batch, original_batches=pre_batches)
        self.policy_builders[self.agent_to_policy[agent_id]].add_batch(post_batch)
    self.agent_builders.clear()
    self.agent_to_policy.clear()