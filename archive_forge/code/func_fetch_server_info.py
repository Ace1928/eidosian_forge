from google.protobuf import message
import requests
from absl import logging
from tensorboard import version
from tensorboard.plugins.scalar import metadata as scalars_metadata
from tensorboard.uploader.proto import server_info_pb2
def fetch_server_info(origin, upload_plugins):
    """Fetches server info from a remote server.

    Args:
      origin: The server with which to communicate. Should be a string
        like "https://tensorboard.dev", including protocol, host, and (if
        needed) port.
      upload_plugins: List of plugins names requested by the user and to be
        verified by the server.

    Returns:
      A `server_info_pb2.ServerInfoResponse` message.

    Raises:
      CommunicationError: Upon failure to connect to or successfully
        communicate with the remote server.
    """
    endpoint = '%s/api/uploader' % origin
    server_info_request = _server_info_request(upload_plugins)
    post_body = server_info_request.SerializeToString()
    logging.info('Requested server info: <%r>', server_info_request)
    try:
        response = requests.post(endpoint, data=post_body, timeout=_REQUEST_TIMEOUT_SECONDS, headers={'User-Agent': 'tensorboard/%s' % version.VERSION})
    except requests.RequestException as e:
        raise CommunicationError('Failed to connect to backend: %s' % e)
    if not response.ok:
        raise CommunicationError('Non-OK status from backend (%d %s): %r' % (response.status_code, response.reason, response.content))
    try:
        return server_info_pb2.ServerInfoResponse.FromString(response.content)
    except message.DecodeError as e:
        raise CommunicationError('Corrupt response from backend (%s): %r' % (e, response.content))