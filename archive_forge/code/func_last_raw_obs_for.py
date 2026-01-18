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
def last_raw_obs_for(self, agent_id: AgentID=_DUMMY_AGENT_ID) -> Optional[EnvObsType]:
    """Returns the last un-preprocessed obs for the specified AgentID.

        Args:
            agent_id: The agent's ID to get the last un-preprocessed
                observation for.

        Returns:
            Last un-preprocessed observation the specified AgentID has seen.
            None in case the agent has never made any observations in the
            episode.
        """
    return self._agent_to_last_raw_obs.get(agent_id)