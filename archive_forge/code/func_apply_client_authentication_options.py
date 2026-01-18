import abc
import base64
import enum
import json
import six
from google.auth import exceptions
def apply_client_authentication_options(self, headers, request_body=None, bearer_token=None):
    """Applies client authentication on the OAuth request's headers or POST
        body.

        Args:
            headers (Mapping[str, str]): The HTTP request header.
            request_body (Optional[Mapping[str, str]]): The HTTP request body
                dictionary. For requests that do not support request body, this
                is None and will be ignored.
            bearer_token (Optional[str]): The optional bearer token.
        """
    self._inject_authenticated_headers(headers, bearer_token)
    if bearer_token is None:
        self._inject_authenticated_request_body(request_body)