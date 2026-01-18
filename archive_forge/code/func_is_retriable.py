from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
import sys
from google.auth import exceptions as auth_exceptions
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage import retry_util
import requests
def is_retriable(exc_type=None, exc_value=None, exc_traceback=None, state=None):
    """Returns True if the error is retriable."""
    del exc_type, exc_traceback, state
    return isinstance(exc_value, (auth_exceptions.TransportError, errors.RetryableApiError, exceptions.BadGateway, exceptions.GatewayTimeout, exceptions.InternalServerError, exceptions.TooManyRequests, exceptions.ServiceUnavailable, requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError, requests.exceptions.Timeout, ConnectionError))