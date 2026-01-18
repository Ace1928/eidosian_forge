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
def image_boxes(tag, tensor_image, tensor_boxes, rescale=1, dataformats='CHW', labels=None):
    """Output a `Summary` protocol buffer with images."""
    tensor_image = make_np(tensor_image)
    tensor_image = convert_to_HWC(tensor_image, dataformats)
    tensor_boxes = make_np(tensor_boxes)
    tensor_image = tensor_image.astype(np.float32) * _calc_scale_factor(tensor_image)
    image = make_image(tensor_image.clip(0, 255).astype(np.uint8), rescale=rescale, rois=tensor_boxes, labels=labels)
    return Summary(value=[Summary.Value(tag=tag, image=image)])