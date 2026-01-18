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
def half_to_int(f: float) -> int:
    """Casts a half-precision float value into an integer.

    Converts a half precision floating point value, such as `torch.half` or
    `torch.bfloat16`, into an integer value which can be written into the
    half_val field of a TensorProto for storage.

    To undo the effects of this conversion, use int_to_half().

    """
    buf = struct.pack('f', f)
    return struct.unpack('i', buf)[0]