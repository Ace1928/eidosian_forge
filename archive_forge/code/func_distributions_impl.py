from werkzeug import wrappers
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.distribution import compressor
from tensorboard.plugins.distribution import metadata
from tensorboard.plugins.histogram import histograms_plugin
def distributions_impl(self, ctx, tag, run, experiment):
    """Result of the form `(body, mime_type)`.

        Raises:
          tensorboard.errors.PublicError: On invalid request.
        """
    histograms, mime_type = self._histograms_plugin.histograms_impl(ctx, tag, run, experiment=experiment, downsample_to=self.SAMPLE_SIZE)
    return ([self._compress(histogram) for histogram in histograms], mime_type)