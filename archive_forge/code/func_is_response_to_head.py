from __future__ import absolute_import
from email.errors import MultipartInvariantViolationDefect, StartBoundaryNotFoundDefect
from ..exceptions import HeaderParsingError
from ..packages.six.moves import http_client as httplib
def is_response_to_head(response):
    """
    Checks whether the request of a response has been a HEAD-request.
    Handles the quirks of AppEngine.

    :param http.client.HTTPResponse response:
        Response to check if the originating request
        used 'HEAD' as a method.
    """
    method = response._method
    if isinstance(method, int):
        return method == 3
    return method.upper() == 'HEAD'