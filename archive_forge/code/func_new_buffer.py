import collections
import platform
import random
from typing import Optional
from ray.util.timer import _Timer
from ray.rllib.execution.replay_ops import SimpleReplayBuffer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, concat_samples
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import ReplayMode
from ray.rllib.utils.replay_buffers.replay_buffer import _ALL_POLICIES
from ray.rllib.utils.typing import PolicyID, SampleBatchType
def new_buffer():
    return SimpleReplayBuffer(num_slots=capacity)