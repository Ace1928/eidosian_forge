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
def start_uploading(self):
    """Uploads data from the logdir.

        This will continuously scan the logdir, uploading as data is added
        unless the uploader was built with the _one_shot option, in which
        case it will terminate after the first scan.

        Raises:
          RuntimeError: If `create_experiment` has not yet been called.
          ExperimentNotFoundError: If the experiment is deleted during the
            course of the upload.
        """
    if self._request_sender is None:
        raise RuntimeError('Must call create_experiment() before start_uploading()')
    while True:
        self._logdir_poll_rate_limiter.tick()
        self._upload_once()
        if self._one_shot:
            break