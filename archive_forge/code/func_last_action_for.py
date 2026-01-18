import random
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.utils.annotations import Deprecated, DeveloperAPI
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.rllib.utils.typing import (
from ray.util import log_once
@DeveloperAPI
def last_action_for(self, agent_id: AgentID=_DUMMY_AGENT_ID) -> EnvActionType:
    """Returns the last action for the specified AgentID, or zeros.

        The "last" action is the most recent one taken by the agent.

        Args:
            agent_id: The agent's ID to get the last action for.

        Returns:
            Last action the specified AgentID has executed.
            Zeros in case the agent has never performed any actions in the
            episode.
        """
    policy_id = self.policy_for(agent_id)
    policy = self.policy_map[policy_id]
    if agent_id in self._agent_to_last_action:
        if policy.config.get('_disable_action_flattening'):
            return self._agent_to_last_action[agent_id]
        else:
            return flatten_to_single_ndarray(self._agent_to_last_action[agent_id])
    elif policy.config.get('_disable_action_flattening'):
        return tree.map_structure(lambda s: np.zeros_like(s.sample(), s.dtype) if hasattr(s, 'dtype') else np.zeros_like(s.sample()), policy.action_space_struct)
    else:
        flat = flatten_to_single_ndarray(policy.action_space.sample())
        if hasattr(policy.action_space, 'dtype'):
            return np.zeros_like(flat, dtype=policy.action_space.dtype)
        return np.zeros_like(flat)