from werkzeug import wrappers
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.distribution import compressor
from tensorboard.plugins.distribution import metadata
from tensorboard.plugins.histogram import histograms_plugin
@wrappers.Request.application
def distributions_route(self, request):
    """Given a tag and single run, return an array of compressed
        histograms."""
    ctx = plugin_util.context(request.environ)
    experiment = plugin_util.experiment_id(request.environ)
    tag = request.args.get('tag')
    run = request.args.get('run')
    body, mime_type = self.distributions_impl(ctx, tag, run, experiment=experiment)
    return http_util.Respond(request, body, mime_type)