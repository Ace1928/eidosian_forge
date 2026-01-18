from gymnasium.spaces import Box, MultiDiscrete, Tuple as TupleSpace
import logging
import numpy as np
import random
import time
from typing import Callable, Optional, Tuple
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import MultiAgentDict, PolicyID, AgentID
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return game_name