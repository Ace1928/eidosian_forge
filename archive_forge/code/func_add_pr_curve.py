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
def add_pr_curve(self, tag, labels, predictions, global_step=None, num_thresholds=127, weights=None, walltime=None):
    """Add precision recall curve.

        Plotting a precision-recall curve lets you understand your model's
        performance under different threshold settings. With this function,
        you provide the ground truth labeling (T/F) and prediction confidence
        (usually the output of your model) for each target. The TensorBoard UI
        will let you choose the threshold interactively.

        Args:
            tag (str): Data identifier
            labels (torch.Tensor, numpy.ndarray, or string/blobname):
              Ground truth data. Binary label for each element.
            predictions (torch.Tensor, numpy.ndarray, or string/blobname):
              The probability that an element be classified as true.
              Value should be in [0, 1]
            global_step (int): Global step value to record
            num_thresholds (int): Number of thresholds used to draw the curve.
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            import numpy as np
            labels = np.random.randint(2, size=100)  # binary label
            predictions = np.random.rand(100)
            writer = SummaryWriter()
            writer.add_pr_curve('pr_curve', labels, predictions, 0)
            writer.close()

        """
    torch._C._log_api_usage_once('tensorboard.logging.add_pr_curve')
    labels, predictions = (make_np(labels), make_np(predictions))
    self._get_file_writer().add_summary(pr_curve(tag, labels, predictions, num_thresholds, weights), global_step, walltime)