import re
from google.protobuf import json_format
from werkzeug import wrappers
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.compat import tf
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorboard.plugins.custom_scalar import metadata
from tensorboard.plugins.scalar import metadata as scalars_metadata
from tensorboard.plugins.scalar import scalars_plugin
@wrappers.Request.application
def scalars_route(self, request):
    """Given a tag regex and single run, return ScalarEvents.

        This route takes 2 GET params:
        run: A run string to find tags for.
        tag: A string that is a regex used to find matching tags.
        The response is a JSON object:
        {
          // Whether the regular expression is valid. Also false if empty.
          regexValid: boolean,

          // An object mapping tag name to a list of ScalarEvents.
          payload: Object<string, ScalarEvent[]>,
        }
        """
    ctx = plugin_util.context(request.environ)
    tag_regex_string = request.args.get('tag')
    run = request.args.get('run')
    experiment = plugin_util.experiment_id(request.environ)
    mime_type = 'application/json'
    try:
        body = self.scalars_impl(ctx, run, tag_regex_string, experiment)
    except ValueError as e:
        return http_util.Respond(request=request, content=str(e), content_type='text/plain', code=400)
    return http_util.Respond(request, body, mime_type)