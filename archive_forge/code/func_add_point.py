import contextlib
import functools
import time
import grpc
from google.protobuf import message
from tensorboard.compat.proto import graph_pb2
from tensorboard.compat.proto import summary_pb2
from tensorboard.compat.proto import types_pb2
from tensorboard.uploader.proto import write_service_pb2
from tensorboard.uploader import logdir_loader
from tensorboard.uploader import upload_tracker
from tensorboard.uploader import util
from tensorboard.backend import process_graph
from tensorboard.backend.event_processing import directory_loader
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.plugins.graph import metadata as graphs_metadata
from tensorboard.util import grpc_util
from tensorboard.util import tb_logging
from tensorboard.util import tensor_util
def add_point(self, point_proto):
    """Integrates the cost of a point proto into the byte budget.

        Args:
          point_proto: The proto representing a point.

        Raises:
          _OutOfSpaceError: If adding the point would exceed the remaining request
           budget.
        """
    submessage_cost = point_proto.ByteSize()
    cost = submessage_cost + _varint_cost(submessage_cost) + 1
    if cost > self._byte_budget:
        raise _OutOfSpaceError()
    self._byte_budget -= cost