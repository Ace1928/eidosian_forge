import copy
import datetime
import json
import cachetools
import six
from six.moves import urllib
from google.auth import _helpers
from google.auth import _service_account_info
from google.auth import crypt
from google.auth import exceptions
import google.auth.credentials
@classmethod
def from_signing_credentials(cls, credentials, **kwargs):
    """Creates a new :class:`google.auth.jwt.OnDemandCredentials` instance
        from an existing :class:`google.auth.credentials.Signing` instance.

        The new instance will use the same signer as the existing instance and
        will use the existing instance's signer email as the issuer and
        subject by default.

        Example::

            svc_creds = service_account.Credentials.from_service_account_file(
                'service_account.json')
            jwt_creds = jwt.OnDemandCredentials.from_signing_credentials(
                svc_creds)

        Args:
            credentials (google.auth.credentials.Signing): The credentials to
                use to construct the new credentials.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.jwt.Credentials: A new Credentials instance.
        """
    kwargs.setdefault('issuer', credentials.signer_email)
    kwargs.setdefault('subject', credentials.signer_email)
    return cls(credentials.signer, **kwargs)