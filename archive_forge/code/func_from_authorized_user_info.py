from datetime import datetime
import io
import json
import logging
import six
from google.auth import _cloud_sdk
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.oauth2 import reauth
@classmethod
def from_authorized_user_info(cls, info, scopes=None):
    """Creates a Credentials instance from parsed authorized user info.

        Args:
            info (Mapping[str, str]): The authorized user info in Google
                format.
            scopes (Sequence[str]): Optional list of scopes to include in the
                credentials.

        Returns:
            google.oauth2.credentials.Credentials: The constructed
                credentials.

        Raises:
            ValueError: If the info is not in the expected format.
        """
    keys_needed = set(('refresh_token', 'client_id', 'client_secret'))
    missing = keys_needed.difference(six.iterkeys(info))
    if missing:
        raise ValueError('Authorized user info was not in the expected format, missing fields {}.'.format(', '.join(missing)))
    expiry = info.get('expiry')
    if expiry:
        expiry = datetime.strptime(expiry.rstrip('Z').split('.')[0], '%Y-%m-%dT%H:%M:%S')
    else:
        expiry = _helpers.utcnow() - _helpers.REFRESH_THRESHOLD
    if scopes is None and 'scopes' in info:
        scopes = info.get('scopes')
        if isinstance(scopes, six.string_types):
            scopes = scopes.split(' ')
    return cls(token=info.get('token'), refresh_token=info.get('refresh_token'), token_uri=_GOOGLE_OAUTH2_TOKEN_ENDPOINT, scopes=scopes, client_id=info.get('client_id'), client_secret=info.get('client_secret'), quota_project_id=info.get('quota_project_id'), expiry=expiry, rapt_token=info.get('rapt_token'))