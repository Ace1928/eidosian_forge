import json
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.backend import process_graph
from tensorboard.compat.proto import config_pb2
from tensorboard.compat.proto import graph_pb2
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.graph import graph_util
from tensorboard.plugins.graph import keras_util
from tensorboard.plugins.graph import metadata
from tensorboard.util import tb_logging
@wrappers.Request.application
def graph_route(self, request):
    """Given a single run, return the graph definition in protobuf
        format."""
    ctx = plugin_util.context(request.environ)
    experiment = plugin_util.experiment_id(request.environ)
    run = request.args.get('run')
    tag = request.args.get('tag')
    conceptual_arg = request.args.get('conceptual', False)
    is_conceptual = True if conceptual_arg == 'true' else False
    if run is None:
        return http_util.Respond(request, 'query parameter "run" is required', 'text/plain', 400)
    limit_attr_size = request.args.get('limit_attr_size', None)
    if limit_attr_size is not None:
        try:
            limit_attr_size = int(limit_attr_size)
        except ValueError:
            return http_util.Respond(request, 'query parameter `limit_attr_size` must be an integer', 'text/plain', 400)
    large_attrs_key = request.args.get('large_attrs_key', None)
    try:
        result = self.graph_impl(ctx, run, tag, is_conceptual, experiment, limit_attr_size, large_attrs_key)
    except ValueError as e:
        return http_util.Respond(request, e.message, 'text/plain', code=400)
    body, mime_type = result
    return http_util.Respond(request, body, mime_type)