import collections
import logging
import random
from typing import Any, Dict, Optional
import numpy as np
from ray.rllib.policy.rnn_sequencing import timeslice_along_seq_lens_with_overlap
from ray.rllib.policy.sample_batch import (
from ray.rllib.utils.annotations import override
from ray.rllib.utils.replay_buffers.multi_agent_prioritized_replay_buffer import (
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import (
from ray.rllib.utils.replay_buffers.replay_buffer import _ALL_POLICIES, StorageUnit
from ray.rllib.utils.typing import PolicyID, SampleBatchType
from ray.util.annotations import DeveloperAPI
from ray.util.debug import log_once
def round_up_or_down(value, ratio):
    """Returns an integer averaging to value*ratio."""
    product = value * ratio
    ceil_prob = product % 1
    if random.uniform(0, 1) < ceil_prob:
        return int(np.ceil(product))
    else:
        return int(np.floor(product))