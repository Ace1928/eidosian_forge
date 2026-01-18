import json
import os
import six
from six.moves import http_client
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import jwt
import google.auth.transport.requests
def fetch_id_token_credentials(audience, request=None):
    """Create the ID Token credentials from the current environment.

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

        # Create ID token credentials.
        credentials = google.oauth2.id_token.fetch_id_token_credentials(target_audience, request=request)

        # Refresh the credential to obtain an ID token.
        credentials.refresh(request)

        id_token = credentials.token
        id_token_expiry = credentials.expiry

    Args:
        audience (str): The audience that this ID token is intended for.
        request (Optional[google.auth.transport.Request]): A callable used to make
            HTTP requests. A request object will be created if not provided.

    Returns:
        google.auth.credentials.Credentials: The ID token credentials.

    Raises:
        ~google.auth.exceptions.DefaultCredentialsError:
            If metadata server doesn't exist and no valid service account
            credentials are found.
    """
    credentials_filename = os.environ.get(environment_vars.CREDENTIALS)
    if credentials_filename:
        if not (os.path.exists(credentials_filename) and os.path.isfile(credentials_filename)):
            raise exceptions.DefaultCredentialsError('GOOGLE_APPLICATION_CREDENTIALS path is either not found or invalid.')
        try:
            with open(credentials_filename, 'r') as f:
                from google.oauth2 import service_account
                info = json.load(f)
                if info.get('type') == 'service_account':
                    return service_account.IDTokenCredentials.from_service_account_info(info, target_audience=audience)
        except ValueError as caught_exc:
            new_exc = exceptions.DefaultCredentialsError('GOOGLE_APPLICATION_CREDENTIALS is not valid service account credentials.', caught_exc)
            six.raise_from(new_exc, caught_exc)
    try:
        from google.auth import compute_engine
        from google.auth.compute_engine import _metadata
        if not request:
            request = google.auth.transport.requests.Request()
        if _metadata.ping(request):
            return compute_engine.IDTokenCredentials(request, audience, use_metadata_identity_endpoint=True)
    except (ImportError, exceptions.TransportError):
        pass
    raise exceptions.DefaultCredentialsError('Neither metadata server or valid service account credentials are found.')