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
@wrappers.Request.application
def list_metric_evals_route(self, request):
    ctx = plugin_util.context(request.environ)
    experiment_id = plugin_util.experiment_id(request.environ)
    try:
        request_proto = _parse_request_argument(request, api_pb2.ListMetricEvalsRequest)
        scalars_plugin = self._get_scalars_plugin()
        if not scalars_plugin:
            raise werkzeug.exceptions.NotFound('Scalars plugin not loaded')
        return http_util.Respond(request, list_metric_evals.Handler(ctx, request_proto, scalars_plugin, experiment_id).run(), 'application/json')
    except error.HParamsError as e:
        logger.error('HParams error: %s' % e)
        raise werkzeug.exceptions.BadRequest(description=str(e))