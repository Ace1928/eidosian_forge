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
def has_pending_agent_data(self) -> bool:
    """Returns whether there is pending unprocessed data.

        Returns:
            bool: True if there is at least one per-agent builder (with data
                in it).
        """
    return len(self.agent_builders) > 0