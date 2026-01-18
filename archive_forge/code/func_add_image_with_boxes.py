import os
import time
from typing import List, Optional, Union, TYPE_CHECKING
import torch
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto.event_pb2 import Event, SessionLog
from tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from ._convert_np import make_np
from ._embedding import get_embedding_info, make_mat, make_sprite, make_tsv, write_pbtxt
from ._onnx_graph import load_onnx_graph
from ._pytorch_graph import graph
from ._utils import figure_to_image
from .summary import (
def add_image_with_boxes(self, tag, img_tensor, box_tensor, global_step=None, walltime=None, rescale=1, dataformats='CHW', labels=None):
    """Add image and draw bounding boxes on the image.

        Args:
            tag (str): Data identifier
            img_tensor (torch.Tensor, numpy.ndarray, or string/blobname): Image data
            box_tensor (torch.Tensor, numpy.ndarray, or string/blobname): Box data (for detected objects)
              box should be represented as [x1, y1, x2, y2].
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
            rescale (float): Optional scale override
            dataformats (str): Image data format specification of the form
              NCHW, NHWC, CHW, HWC, HW, WH, etc.
            labels (list of string): The label to be shown for each bounding box.
        Shape:
            img_tensor: Default is :math:`(3, H, W)`. It can be specified with ``dataformats`` argument.
            e.g. CHW or HWC

            box_tensor: (torch.Tensor, numpy.ndarray, or string/blobname): NX4,  where N is the number of
            boxes and each 4 elements in a row represents (xmin, ymin, xmax, ymax).
        """
    torch._C._log_api_usage_once('tensorboard.logging.add_image_with_boxes')
    if self._check_caffe2_blob(img_tensor):
        from caffe2.python import workspace
        img_tensor = workspace.FetchBlob(img_tensor)
    if self._check_caffe2_blob(box_tensor):
        from caffe2.python import workspace
        box_tensor = workspace.FetchBlob(box_tensor)
    if labels is not None:
        if isinstance(labels, str):
            labels = [labels]
        if len(labels) != box_tensor.shape[0]:
            labels = None
    self._get_file_writer().add_summary(image_boxes(tag, img_tensor, box_tensor, rescale=rescale, dataformats=dataformats, labels=labels), global_step, walltime)