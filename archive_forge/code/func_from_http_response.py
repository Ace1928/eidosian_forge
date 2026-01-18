from __future__ import absolute_import
from __future__ import unicode_literals
import http.client
from typing import Dict
from typing import Union
import warnings
from google.rpc import error_details_pb2
def from_http_response(response):
    """Create a :class:`GoogleAPICallError` from a :class:`requests.Response`.

    Args:
        response (requests.Response): The HTTP response.

    Returns:
        GoogleAPICallError: An instance of the appropriate subclass of
            :class:`GoogleAPICallError`, with the message and errors populated
            from the response.
    """
    try:
        payload = response.json()
    except ValueError:
        payload = {'error': {'message': response.text or 'unknown error'}}
    error_message = payload.get('error', {}).get('message', 'unknown error')
    errors = payload.get('error', {}).get('errors', ())
    details = payload.get('error', {}).get('details', ())
    error_info = list(filter(lambda detail: detail.get('@type', '') == 'type.googleapis.com/google.rpc.ErrorInfo', details))
    error_info = error_info[0] if error_info else None
    message = '{method} {url}: {error}'.format(method=response.request.method, url=response.request.url, error=error_message)
    exception = from_http_status(response.status_code, message, errors=errors, details=details, response=response, error_info=error_info)
    return exception