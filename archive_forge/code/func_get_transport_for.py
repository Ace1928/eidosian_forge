import urllib.parse
from stevedore import driver
from zaqarclient import errors as _errors
def get_transport_for(url_or_request, version=1, options=None):
    """Gets a transport for a given url.

    An example transport URL might be::

        zmq://example.org:8888/v1/

    :param url_or_request: a transport URL
    :type url_or_request: str or
        `zaqarclient.transport.request.Request`
    :param version: Version of the target transport.
    :type version: int

    :returns: A `Transport` instance.
    :rtype: `zaqarclient.transport.Transport`
    """
    url = url_or_request
    if not isinstance(url_or_request, str):
        url = url_or_request.endpoint
    parsed = urllib.parse.urlparse(url)
    return get_transport(parsed.scheme, version, options)