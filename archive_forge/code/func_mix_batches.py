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
def mix_batches(_policy_id):
    """Mixes old with new samples.

            Tries to mix according to self.replay_ratio on average.
            If not enough new samples are available, mixes in less old samples
            to retain self.replay_ratio on average.
            """

    def round_up_or_down(value, ratio):
        """Returns an integer averaging to value*ratio."""
        product = value * ratio
        ceil_prob = product % 1
        if random.uniform(0, 1) < ceil_prob:
            return int(np.ceil(product))
        else:
            return int(np.floor(product))
    max_num_new = round_up_or_down(num_items, 1 - self.replay_ratio)
    _buffer = self.replay_buffers[_policy_id]
    output_batches = self.last_added_batches[_policy_id][:max_num_new]
    self.last_added_batches[_policy_id] = self.last_added_batches[_policy_id][max_num_new:]
    if self.replay_ratio == 0.0:
        return concat_samples_into_ma_batch(output_batches)
    elif self.replay_ratio == 1.0:
        return _buffer.sample(num_items, **kwargs)
    num_new = len(output_batches)
    if np.isclose(num_new, num_items * (1 - self.replay_ratio)):
        num_old = num_items - max_num_new
    else:
        num_old = min(num_items - max_num_new, round_up_or_down(num_new, self.replay_ratio / (1 - self.replay_ratio)))
    output_batches.append(_buffer.sample(num_old, **kwargs))
    output_batches = [batch.as_multi_agent() for batch in output_batches]
    return concat_samples_into_ma_batch(output_batches)