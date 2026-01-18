import os.path
import re
from .. import exceptions as exc
def get_download_file_path(response, path):
    """
    Given a response and a path, return a file path for a download.

    If a ``path`` parameter is a directory, this function will parse the
    ``Content-Disposition`` header on the response to determine the name of the
    file as reported by the server, and return a file path in the specified
    directory.

    If ``path`` is empty or None, this function will return a path relative
    to the process' current working directory.

    If path is a full file path, return it.

    :param response: A Response object from requests
    :type response: requests.models.Response
    :param str path: Directory or file path.
    :returns: full file path to download as
    :rtype: str
    :raises: :class:`requests_toolbelt.exceptions.StreamingError`
    """
    path_is_dir = path and os.path.isdir(path)
    if path and (not path_is_dir):
        filepath = path
    else:
        response_filename = _get_filename(response.headers.get('content-disposition', ''))
        if not response_filename:
            raise exc.StreamingError('No filename given to stream response to')
        if path_is_dir:
            filepath = os.path.join(path, response_filename)
        else:
            filepath = response_filename
    return filepath