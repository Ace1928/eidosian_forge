import json
import os
import six
from six.moves import http_client
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import jwt
import google.auth.transport.requests
def fetch_id_token(request, audience):
    """Fetch the ID Token from the current environment.

    This function acquires ID token from the environment in the following order.
    See https://google.aip.dev/auth/4110.

    1. If the environment variable ``GOOGLE_APPLICATION_CREDENTIALS`` is set
       to the path of a valid service account JSON file, then ID token is
       acquired using this service account credentials.
    2. If the application is running in Compute Engine, App Engine or Cloud Run,
       then the ID token are obtained from the metadata server.
    3. If metadata server doesn't exist and no valid service account credentials
       are found, :class:`~google.auth.exceptions.DefaultCredentialsError` will
       be raised.

    Example::

        import google.oauth2.id_token
        import google.auth.transport.requests

        request = google.auth.transport.requests.Request()
        target_audience = "https://pubsub.googleapis.com"

        id_token = google.oauth2.id_token.fetch_id_token(request, target_audience)

    Args:
        request (google.auth.transport.Request): A callable used to make
            HTTP requests.
        audience (str): The audience that this ID token is intended for.

    Returns:
        str: The ID token.

    Raises:
        ~google.auth.exceptions.DefaultCredentialsError:
            If metadata server doesn't exist and no valid service account
            credentials are found.
    """
    id_token_credentials = fetch_id_token_credentials(audience, request=request)
    id_token_credentials.refresh(request)
    return id_token_credentials.token