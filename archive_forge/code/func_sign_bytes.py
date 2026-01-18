import datetime
import six
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.auth import iam
from google.auth import jwt
from google.auth.compute_engine import _metadata
from google.oauth2 import _client
def sign_bytes(self, message):
    """Signs the given message.

        Args:
            message (bytes): The message to sign.

        Returns:
            bytes: The message's cryptographic signature.

        Raises:
            ValueError:
                Signer is not available if metadata identity endpoint is used.
        """
    if self._use_metadata_identity_endpoint:
        raise exceptions.InvalidOperation('Signer is not available if metadata identity endpoint is used')
    return self._signer.sign(message)