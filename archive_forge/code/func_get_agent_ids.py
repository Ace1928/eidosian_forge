import gymnasium as gym
import logging
from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import (
from ray.rllib.utils.typing import (
from ray.util import log_once
@override(BaseEnv)
def get_agent_ids(self) -> Set[AgentID]:
    return self.envs[0].get_agent_ids()