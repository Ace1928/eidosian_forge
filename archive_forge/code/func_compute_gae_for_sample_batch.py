import numpy as np
import scipy.signal
from typing import Dict, Optional
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import AgentID
from ray.rllib.utils.typing import TensorType
@DeveloperAPI
def compute_gae_for_sample_batch(policy: Policy, sample_batch: SampleBatch, other_agent_batches: Optional[Dict[AgentID, SampleBatch]]=None, episode: Optional[Episode]=None) -> SampleBatch:
    """Adds GAE (generalized advantage estimations) to a trajectory.

    The trajectory contains only data from one episode and from one agent.
    - If  `config.batch_mode=truncate_episodes` (default), sample_batch may
    contain a truncated (at-the-end) episode, in case the
    `config.rollout_fragment_length` was reached by the sampler.
    - If `config.batch_mode=complete_episodes`, sample_batch will contain
    exactly one episode (no matter how long).
    New columns can be added to sample_batch and existing ones may be altered.

    Args:
        policy: The Policy used to generate the trajectory (`sample_batch`)
        sample_batch: The SampleBatch to postprocess.
        other_agent_batches: Optional dict of AgentIDs mapping to other
            agents' trajectory data (from the same episode).
            NOTE: The other agents use the same policy.
        episode: Optional multi-agent episode object in which the agents
            operated.

    Returns:
        The postprocessed, modified SampleBatch (or a new one).
    """
    sample_batch = compute_bootstrap_value(sample_batch, policy)
    vf_preds = np.array(sample_batch[SampleBatch.VF_PREDS])
    rewards = np.array(sample_batch[SampleBatch.REWARDS])
    if len(vf_preds.shape) == 2:
        assert vf_preds.shape == rewards.shape
        vf_preds = np.squeeze(vf_preds, axis=1)
        rewards = np.squeeze(rewards, axis=1)
        squeezed = True
    else:
        squeezed = False
    batch = compute_advantages(rollout=sample_batch, last_r=sample_batch[SampleBatch.VALUES_BOOTSTRAPPED][-1], gamma=policy.config['gamma'], lambda_=policy.config['lambda'], use_gae=policy.config['use_gae'], use_critic=policy.config.get('use_critic', True), vf_preds=vf_preds, rewards=rewards)
    if squeezed:
        batch[Postprocessing.ADVANTAGES] = np.expand_dims(batch[Postprocessing.ADVANTAGES], axis=1)
    return batch