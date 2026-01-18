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
def compute_curve(labels, predictions, num_thresholds=None, weights=None):
    _MINIMUM_COUNT = 1e-07
    if weights is None:
        weights = 1.0
    bucket_indices = np.int32(np.floor(predictions * (num_thresholds - 1)))
    float_labels = labels.astype(np.float64)
    histogram_range = (0, num_thresholds - 1)
    tp_buckets, _ = np.histogram(bucket_indices, bins=num_thresholds, range=histogram_range, weights=float_labels * weights)
    fp_buckets, _ = np.histogram(bucket_indices, bins=num_thresholds, range=histogram_range, weights=(1.0 - float_labels) * weights)
    tp = np.cumsum(tp_buckets[::-1])[::-1]
    fp = np.cumsum(fp_buckets[::-1])[::-1]
    tn = fp[0] - fp
    fn = tp[0] - tp
    precision = tp / np.maximum(_MINIMUM_COUNT, tp + fp)
    recall = tp / np.maximum(_MINIMUM_COUNT, tp + fn)
    return np.stack((tp, fp, tn, fn, precision, recall))