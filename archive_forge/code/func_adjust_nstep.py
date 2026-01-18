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
def adjust_nstep(n_step: int, gamma: float, batch: SampleBatch) -> None:
    """Rewrites `batch` to encode n-step rewards, terminateds, truncateds, and next-obs.

    Observations and actions remain unaffected. At the end of the trajectory,
    n is truncated to fit in the traj length.

    Args:
        n_step: The number of steps to look ahead and adjust.
        gamma: The discount factor.
        batch: The SampleBatch to adjust (in place).

    Examples:
        n-step=3
        Trajectory=o0 r0 d0, o1 r1 d1, o2 r2 d2, o3 r3 d3, o4 r4 d4=True o5
        gamma=0.9
        Returned trajectory:
        0: o0 [r0 + 0.9*r1 + 0.9^2*r2 + 0.9^3*r3] d3 o0'=o3
        1: o1 [r1 + 0.9*r2 + 0.9^2*r3 + 0.9^3*r4] d4 o1'=o4
        2: o2 [r2 + 0.9*r3 + 0.9^2*r4] d4 o1'=o5
        3: o3 [r3 + 0.9*r4] d4 o3'=o5
        4: o4 r4 d4 o4'=o5
    """
    assert batch.is_single_trajectory(), 'Unexpected terminated|truncated in middle of trajectory!'
    len_ = len(batch)
    batch[SampleBatch.NEXT_OBS] = np.concatenate([batch[SampleBatch.OBS][n_step:], np.stack([batch[SampleBatch.NEXT_OBS][-1]] * min(n_step, len_))], axis=0)
    batch[SampleBatch.TERMINATEDS] = np.concatenate([batch[SampleBatch.TERMINATEDS][n_step - 1:], np.tile(batch[SampleBatch.TERMINATEDS][-1], min(n_step - 1, len_))], axis=0)
    if SampleBatch.TRUNCATEDS in batch:
        batch[SampleBatch.TRUNCATEDS] = np.concatenate([batch[SampleBatch.TRUNCATEDS][n_step - 1:], np.tile(batch[SampleBatch.TRUNCATEDS][-1], min(n_step - 1, len_))], axis=0)
    for i in range(len_):
        for j in range(1, n_step):
            if i + j < len_:
                batch[SampleBatch.REWARDS][i] += gamma ** j * batch[SampleBatch.REWARDS][i + j]