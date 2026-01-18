from collections import defaultdict
from collections.abc import Iterable
import numpy as np
import torch
import hypothesis
from functools import reduce
from hypothesis import assume
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as stnp
from hypothesis.strategies import SearchStrategy
from torch.testing._internal.common_quantized import _calculate_dynamic_qparams, _calculate_dynamic_per_channel_qparams
def assume_not_overflowing(tensor, qparams):
    min_value, max_value = _get_valid_min_max(qparams)
    assume(tensor.min() >= min_value)
    assume(tensor.max() <= max_value)
    return True