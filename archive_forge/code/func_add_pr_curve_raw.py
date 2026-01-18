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
def add_pr_curve_raw(self, tag, true_positive_counts, false_positive_counts, true_negative_counts, false_negative_counts, precision, recall, global_step=None, num_thresholds=127, weights=None, walltime=None):
    """Add precision recall curve with raw data.

        Args:
            tag (str): Data identifier
            true_positive_counts (torch.Tensor, numpy.ndarray, or string/blobname): true positive counts
            false_positive_counts (torch.Tensor, numpy.ndarray, or string/blobname): false positive counts
            true_negative_counts (torch.Tensor, numpy.ndarray, or string/blobname): true negative counts
            false_negative_counts (torch.Tensor, numpy.ndarray, or string/blobname): false negative counts
            precision (torch.Tensor, numpy.ndarray, or string/blobname): precision
            recall (torch.Tensor, numpy.ndarray, or string/blobname): recall
            global_step (int): Global step value to record
            num_thresholds (int): Number of thresholds used to draw the curve.
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
            see: https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/pr_curve/README.md
        """
    torch._C._log_api_usage_once('tensorboard.logging.add_pr_curve_raw')
    self._get_file_writer().add_summary(pr_curve_raw(tag, true_positive_counts, false_positive_counts, true_negative_counts, false_negative_counts, precision, recall, num_thresholds, weights), global_step, walltime)