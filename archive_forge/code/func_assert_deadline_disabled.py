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
def assert_deadline_disabled():
    if hypothesis_version < (3, 27, 0):
        import warnings
        warning_message = f'Your version of hypothesis is outdated. To avoid `DeadlineExceeded` errors, please update. Current hypothesis version: {hypothesis.__version__}'
        warnings.warn(warning_message)
    else:
        assert settings().deadline is None