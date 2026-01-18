import collections
import logging
from enum import Enum
from typing import Any, Dict, Optional
from ray.util.timer import _Timer
from ray.rllib.policy.rnn_sequencing import timeslice_along_seq_lens_with_overlap
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.replay_buffers.replay_buffer import (
from ray.rllib.utils.typing import PolicyID, SampleBatchType
from ray.util.annotations import DeveloperAPI
from ray.util.debug import log_once
@DeveloperAPI
def merge_dicts_with_warning(args_on_init, args_on_call):
    """Merge argument dicts, overwriting args_on_call with warning.

    The MultiAgentReplayBuffer supports setting standard arguments for calls
    of methods of the underlying buffers. These arguments can be
    overwritten. Such overwrites trigger a warning to the user.
    """
    for arg_name, arg_value in args_on_call.items():
        if arg_name in args_on_init:
            if log_once('overwrite_argument_{}'.format(str(arg_name))):
                logger.warning('Replay Buffer was initialized to have underlying buffers methods called with argument `{}={}`, but was subsequently called with `{}={}`.'.format(arg_name, args_on_init[arg_name], arg_name, arg_value))
    return {**args_on_init, **args_on_call}