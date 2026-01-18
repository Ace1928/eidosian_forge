import random
from collections import defaultdict
import numpy as np
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.collectors.simple_list_collector import (
from ray.rllib.evaluation.collectors.agent_collector import AgentCollector
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.typing import AgentID, EnvID, EnvInfoDict, PolicyID, TensorType
@DeveloperAPI
def get_agents(self) -> List[AgentID]:
    """Returns list of agent IDs that have appeared in this episode.

        Returns:
            The list of all agent IDs that have appeared so far in this
            episode.
        """
    return list(self._agent_to_index.keys())