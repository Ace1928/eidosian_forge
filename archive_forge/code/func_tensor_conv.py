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
def tensor_conv(draw, spatial_dim=2, batch_size_range=(1, 4), input_channels_per_group_range=(3, 7), output_channels_per_group_range=(3, 7), feature_map_range=(6, 12), kernel_range=(3, 7), max_groups=1, can_be_transposed=False, elements=None, qparams=None):
    batch_size = draw(st.integers(*batch_size_range))
    input_channels_per_group = draw(st.integers(*input_channels_per_group_range))
    output_channels_per_group = draw(st.integers(*output_channels_per_group_range))
    groups = draw(st.integers(1, max_groups))
    input_channels = input_channels_per_group * groups
    output_channels = output_channels_per_group * groups
    if isinstance(spatial_dim, Iterable):
        spatial_dim = draw(st.sampled_from(spatial_dim))
    feature_map_shape = []
    for i in range(spatial_dim):
        feature_map_shape.append(draw(st.integers(*feature_map_range)))
    kernels = []
    for i in range(spatial_dim):
        kernels.append(draw(st.integers(*kernel_range)))
    tr = False
    weight_shape = (output_channels, input_channels_per_group) + tuple(kernels)
    bias_shape = output_channels
    if can_be_transposed:
        tr = draw(st.booleans())
        if tr:
            weight_shape = (input_channels, output_channels_per_group) + tuple(kernels)
            bias_shape = output_channels
    if qparams is not None:
        if isinstance(qparams, (list, tuple)):
            assert len(qparams) == 3, 'Need 3 qparams for X, w, b'
        else:
            qparams = [qparams] * 3
    X = draw(tensor(shapes=((batch_size, input_channels) + tuple(feature_map_shape),), elements=elements, qparams=qparams[0]))
    W = draw(tensor(shapes=(weight_shape,), elements=elements, qparams=qparams[1]))
    b = draw(tensor(shapes=(bias_shape,), elements=elements, qparams=qparams[2]))
    return (X, W, b, groups, tr)