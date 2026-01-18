from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.histogram import metadata
@wrappers.Request.application
def histograms_route(self, request):
    """Given a tag and single run, return array of histogram values."""
    ctx = plugin_util.context(request.environ)
    experiment = plugin_util.experiment_id(request.environ)
    tag = request.args.get('tag')
    run = request.args.get('run')
    body, mime_type = self.histograms_impl(ctx, tag, run, experiment=experiment, downsample_to=self.SAMPLE_SIZE)
    return http_util.Respond(request, body, mime_type)