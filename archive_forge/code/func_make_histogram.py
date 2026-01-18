import json
import logging
import os
import struct
from typing import Any, List, Optional
import torch
import numpy as np
from google.protobuf import struct_pb2
from tensorboard.compat.proto.summary_pb2 import (
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorboard.plugins.pr_curve.plugin_data_pb2 import PrCurvePluginData
from tensorboard.plugins.text.plugin_data_pb2 import TextPluginData
from ._convert_np import make_np
from ._utils import _prepare_video, convert_to_HWC
def make_histogram(values, bins, max_bins=None):
    """Convert values into a histogram proto using logic from histogram.cc."""
    if values.size == 0:
        raise ValueError('The input has no element.')
    values = values.reshape(-1)
    counts, limits = np.histogram(values, bins=bins)
    num_bins = len(counts)
    if max_bins is not None and num_bins > max_bins:
        subsampling = num_bins // max_bins
        subsampling_remainder = num_bins % subsampling
        if subsampling_remainder != 0:
            counts = np.pad(counts, pad_width=[[0, subsampling - subsampling_remainder]], mode='constant', constant_values=0)
        counts = counts.reshape(-1, subsampling).sum(axis=-1)
        new_limits = np.empty((counts.size + 1,), limits.dtype)
        new_limits[:-1] = limits[:-1:subsampling]
        new_limits[-1] = limits[-1]
        limits = new_limits
    cum_counts = np.cumsum(np.greater(counts, 0))
    start, end = np.searchsorted(cum_counts, [0, cum_counts[-1] - 1], side='right')
    start = int(start)
    end = int(end) + 1
    del cum_counts
    counts = counts[start - 1:end] if start > 0 else np.concatenate([[0], counts[:end]])
    limits = limits[start:end + 1]
    if counts.size == 0 or limits.size == 0:
        raise ValueError('The histogram is empty, please file a bug report.')
    sum_sq = values.dot(values)
    return HistogramProto(min=values.min(), max=values.max(), num=len(values), sum=values.sum(), sum_squares=sum_sq, bucket_limit=limits.tolist(), bucket=counts.tolist())