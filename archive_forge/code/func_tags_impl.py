import numpy as np
from werkzeug import wrappers
from tensorboard import plugin_util
from tensorboard.data import provider
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.pr_curve import metadata
def tags_impl(self, ctx, experiment):
    """Creates the JSON object for the tags route response.

        Returns:
          The JSON object for the tags route response.
        """
    mapping = self._data_provider.list_tensors(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME)
    result = {run: {} for run in mapping}
    for run, tag_to_time_series in mapping.items():
        for tag, time_series in tag_to_time_series.items():
            md = metadata.parse_plugin_metadata(time_series.plugin_content)
            if not self._version_checker.ok(md.version, run, tag):
                continue
            result[run][tag] = {'displayName': time_series.display_name, 'description': plugin_util.markdown_to_safe_html(time_series.description)}
    return result