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
def create_experiment(self):
    """Creates an Experiment for this upload session and returns the ID."""
    logger.info('Creating experiment')
    request = write_service_pb2.CreateExperimentRequest(name=self._name, description=self._description)
    response = grpc_util.call_with_retries(self._api.CreateExperiment, request)
    self._request_sender = _BatchedRequestSender(response.experiment_id, self._api, allowed_plugins=self._allowed_plugins, upload_limits=self._upload_limits, rpc_rate_limiter=self._rpc_rate_limiter, tensor_rpc_rate_limiter=self._tensor_rpc_rate_limiter, blob_rpc_rate_limiter=self._blob_rpc_rate_limiter, tracker=self._tracker)
    self._experiment_id = response.experiment_id
    return response.experiment_id