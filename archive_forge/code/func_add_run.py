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
def add_run(self, run_proto):
    """Integrates the cost of a run proto into the byte budget.

        Args:
          run_proto: The proto representing a run.

        Raises:
          _OutOfSpaceError: If adding the run would exceed the remaining request
            budget.
        """
    cost = run_proto.ByteSize() + _MAX_VARINT64_LENGTH_BYTES + 1
    if cost > self._byte_budget:
        raise _OutOfSpaceError()
    self._byte_budget -= cost