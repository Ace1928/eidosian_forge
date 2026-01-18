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
@st.composite
def qparams(draw, dtypes=None, scale_min=None, scale_max=None, zero_point_min=None, zero_point_max=None):
    if dtypes is None:
        dtypes = _ALL_QINT_TYPES
    if not isinstance(dtypes, (list, tuple)):
        dtypes = (dtypes,)
    quantized_type = draw(st.sampled_from(dtypes))
    _type_info = torch.iinfo(quantized_type)
    qmin, qmax = (_type_info.min, _type_info.max)
    _zp_enforced = _ENFORCED_ZERO_POINT[quantized_type]
    if _zp_enforced is not None:
        zero_point = _zp_enforced
    else:
        _zp_min = qmin if zero_point_min is None else zero_point_min
        _zp_max = qmax if zero_point_max is None else zero_point_max
        zero_point = draw(st.integers(min_value=_zp_min, max_value=_zp_max))
    if scale_min is None:
        scale_min = torch.finfo(torch.float).eps
    if scale_max is None:
        scale_max = torch.finfo(torch.float).max
    scale = draw(floats(min_value=scale_min, max_value=scale_max, width=32))
    return (scale, zero_point, quantized_type)