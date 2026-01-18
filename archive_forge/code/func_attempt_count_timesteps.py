import collections
from functools import partial
import itertools
import sys
from numbers import Number
from typing import Dict, Iterator, Set, Union
from typing import List, Optional
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.utils.annotations import DeveloperAPI, ExperimentalAPI, PublicAPI
from ray.rllib.utils.compression import pack, unpack, is_compressed
from ray.rllib.utils.deprecation import Deprecated, deprecation_warning
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import (
from ray.util import log_once
@DeveloperAPI
def attempt_count_timesteps(tensor_dict: dict):
    """Attempt to count timesteps based on dimensions of individual elements.

    Returns the first successfully counted number of timesteps.
    We do not attempt to count on INFOS or any state_in_* and state_out_* keys. The
    number of timesteps we count in cases where we are unable to count is zero.

    Args:
        tensor_dict: A SampleBatch or another dict.

    Returns:
        count: The inferred number of timesteps >= 0.
    """
    seq_lens = tensor_dict.get(SampleBatch.SEQ_LENS)
    if seq_lens is not None and (not (tf and tf.is_tensor(seq_lens) and (not hasattr(seq_lens, 'numpy')))) and (len(seq_lens) > 0):
        if torch and torch.is_tensor(seq_lens):
            return seq_lens.sum().item()
        else:
            return int(sum(seq_lens))
    for k, v in tensor_dict.items():
        if k == SampleBatch.SEQ_LENS:
            continue
        assert isinstance(k, str), tensor_dict
        if k == SampleBatch.INFOS or k.startswith('state_in_') or k.startswith('state_out_'):
            continue
        v_list = tree.flatten(v) if isinstance(v, (dict, tuple)) else [v]
        v_list = [np.array(_v) if isinstance(_v, (Number, list)) else _v for _v in v_list]
        try:
            _len = len(v_list[0])
            if _len:
                return _len
        except Exception:
            pass
    return 0