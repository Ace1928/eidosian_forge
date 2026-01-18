import numpy as np
from werkzeug import wrappers
from tensorboard import plugin_util
from tensorboard.data import provider
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.pr_curve import metadata
def pr_curves_impl(self, ctx, experiment, runs, tag):
    """Creates the JSON object for the PR curves response for a run-tag
        combo.

        Arguments:
          runs: A list of runs to fetch the curves for.
          tag: The tag to fetch the curves for.

        Raises:
          ValueError: If no PR curves could be fetched for a run and tag.

        Returns:
          The JSON object for the PR curves route response.
        """
    response_mapping = {}
    rtf = provider.RunTagFilter(runs, [tag])
    read_result = self._data_provider.read_tensors(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME, run_tag_filter=rtf, downsample=self._downsample_to)
    for run in runs:
        data = read_result.get(run, {}).get(tag)
        if data is None:
            raise ValueError('No PR curves could be found for run %r and tag %r' % (run, tag))
        response_mapping[run] = [self._process_datum(d) for d in data]
    return response_mapping