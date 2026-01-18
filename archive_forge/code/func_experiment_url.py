from google.protobuf import message
import requests
from absl import logging
from tensorboard import version
from tensorboard.plugins.scalar import metadata as scalars_metadata
from tensorboard.uploader.proto import server_info_pb2
def experiment_url(server_info, experiment_id):
    """Formats a URL that will resolve to the provided experiment.

    Args:
      server_info: A `server_info_pb2.ServerInfoResponse` message.
      experiment_id: A string; the ID of the experiment to link to.

    Returns:
      A URL resolving to the given experiment, as a string.
    """
    url_format = server_info.url_format
    return url_format.template.replace(url_format.id_placeholder, experiment_id)