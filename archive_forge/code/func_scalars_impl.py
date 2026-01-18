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
def scalars_impl(self, ctx, run, tag_regex_string, experiment):
    """Given a tag regex and single run, return ScalarEvents.

        Args:
          ctx: A tensorboard.context.RequestContext value.
          run: A run string.
          tag_regex_string: A regular expression that captures portions of tags.

        Raises:
          ValueError: if the scalars plugin is not registered.

        Returns:
          A dictionary that is the JSON-able response.
        """
    if not tag_regex_string:
        return {_REGEX_VALID_PROPERTY: False, _TAG_TO_EVENTS_PROPERTY: {}}
    try:
        regex = re.compile(tag_regex_string)
    except re.error:
        return {_REGEX_VALID_PROPERTY: False, _TAG_TO_EVENTS_PROPERTY: {}}
    run_to_data = self._data_provider.list_scalars(ctx, experiment_id=experiment, plugin_name=scalars_metadata.PLUGIN_NAME, run_tag_filter=provider.RunTagFilter(runs=[run]))
    tag_to_data = None
    try:
        tag_to_data = run_to_data[run]
    except KeyError:
        payload = {}
    if tag_to_data:
        scalars_plugin_instance = self._get_scalars_plugin()
        if not scalars_plugin_instance:
            raise ValueError('Failed to respond to request for /scalars. The scalars plugin is oddly not registered.')
        form = scalars_plugin.OutputFormat.JSON
        payload = {tag: scalars_plugin_instance.scalars_impl(ctx, tag, run, experiment, form)[0] for tag in tag_to_data.keys() if regex.match(tag)}
    return {_REGEX_VALID_PROPERTY: True, _TAG_TO_EVENTS_PROPERTY: payload}