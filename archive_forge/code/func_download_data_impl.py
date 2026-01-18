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
def download_data_impl(self, ctx, run, tag, experiment, response_format):
    """Provides a response for downloading scalars data for a data series.

        Args:
          ctx: A tensorboard.context.RequestContext value.
          run: The run.
          tag: The specific tag.
          experiment: An experiment ID, as a possibly-empty `str`.
          response_format: A string. One of the values of the OutputFormat enum
            of the scalar plugin.

        Raises:
          ValueError: If the scalars plugin is not registered.

        Returns:
          2 entities:
            - A JSON object response body.
            - A mime type (string) for the response.
        """
    scalars_plugin_instance = self._get_scalars_plugin()
    if not scalars_plugin_instance:
        raise ValueError('Failed to respond to request for /download_data. The scalars plugin is oddly not registered.')
    body, mime_type = scalars_plugin_instance.scalars_impl(ctx, tag, run, experiment, response_format)
    return (body, mime_type)