import logging
import gymnasium as gym
import numpy as np
from typing import Callable, List, Optional, Tuple, Union, Set
from ray.rllib.env.base_env import BaseEnv, _DUMMY_AGENT_ID
from ray.rllib.utils.annotations import Deprecated, override, PublicAPI
from ray.rllib.utils.typing import (
from ray.util import log_once
@override(VectorEnv)
def restart_at(self, index: Optional[int]=None) -> None:
    if index is None:
        index = 0
    try:
        self.envs[index].close()
    except Exception as e:
        if log_once('close_sub_env'):
            logger.warning(f'Trying to close old and replaced sub-environment (at vector index={index}), but closing resulted in error:\n{e}')
    logger.warning(f'Trying to restart sub-environment at index {index}.')
    self.envs[index] = self.make_env(index)
    logger.warning(f'Sub-environment at index {index} restarted successfully.')