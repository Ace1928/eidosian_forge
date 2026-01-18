from typing import Tuple, Union
import numpy as np
from gym.core import ObsType
def convert_to_terminated_truncated_step_api(step_returns: Union[DoneStepType, TerminatedTruncatedStepType], is_vector_env=False) -> TerminatedTruncatedStepType:
    """Function to transform step returns to new step API irrespective of input API.

    Args:
        step_returns (tuple): Items returned by step(). Can be (obs, rew, done, info) or (obs, rew, terminated, truncated, info)
        is_vector_env (bool): Whether the step_returns are from a vector environment
    """
    if len(step_returns) == 5:
        return step_returns
    else:
        assert len(step_returns) == 4
        observations, rewards, dones, infos = step_returns
        if is_vector_env is False:
            truncated = infos.pop('TimeLimit.truncated', False)
            return (observations, rewards, dones and (not truncated), dones and truncated, infos)
        elif isinstance(infos, list):
            truncated = np.array([info.pop('TimeLimit.truncated', False) for info in infos])
            return (observations, rewards, np.logical_and(dones, np.logical_not(truncated)), np.logical_and(dones, truncated), infos)
        elif isinstance(infos, dict):
            num_envs = len(dones)
            truncated = infos.pop('TimeLimit.truncated', np.zeros(num_envs, dtype=bool))
            return (observations, rewards, np.logical_and(dones, np.logical_not(truncated)), np.logical_and(dones, truncated), infos)
        else:
            raise TypeError(f'Unexpected value of infos, as is_vector_envs=False, expects `info` to be a list or dict, actual type: {type(infos)}')