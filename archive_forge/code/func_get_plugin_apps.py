import json
import werkzeug
from werkzeug import wrappers
from tensorboard import plugin_util
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import backend_context
from tensorboard.plugins.hparams import download_data
from tensorboard.plugins.hparams import error
from tensorboard.plugins.hparams import get_experiment
from tensorboard.plugins.hparams import list_metric_evals
from tensorboard.plugins.hparams import list_session_groups
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.scalar import metadata as scalars_metadata
from tensorboard.util import tb_logging
def get_plugin_apps(self):
    """See base class."""
    return {'/download_data': self.download_data_route, '/experiment': self.get_experiment_route, '/session_groups': self.list_session_groups_route, '/metric_evals': self.list_metric_evals_route}