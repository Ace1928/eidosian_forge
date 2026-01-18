from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.histogram import metadata
def histograms_impl(self, ctx, tag, run, experiment, downsample_to=None):
    """Result of the form `(body, mime_type)`.

        At most `downsample_to` events will be returned. If this value is
        `None`, then default downsampling will be performed.

        Raises:
          tensorboard.errors.PublicError: On invalid request.
        """
    sample_count = downsample_to if downsample_to is not None else self._downsample_to
    all_histograms = self._data_provider.read_tensors(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME, downsample=sample_count, run_tag_filter=provider.RunTagFilter(runs=[run], tags=[tag]))
    histograms = all_histograms.get(run, {}).get(tag, None)
    if histograms is None:
        raise errors.NotFoundError('No histogram tag %r for run %r' % (tag, run))
    events = [(e.wall_time, e.step, e.numpy.tolist()) for e in histograms]
    return (events, 'application/json')