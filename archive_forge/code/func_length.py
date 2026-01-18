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
@property
def length(self):
    return self.total_env_steps