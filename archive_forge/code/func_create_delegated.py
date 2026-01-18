import base64
import copy
import datetime
import json
import time
import oauth2client
from oauth2client import _helpers
from oauth2client import client
from oauth2client import crypt
from oauth2client import transport
def create_delegated(self, sub):
    """Create credentials that act as domain-wide delegation of authority.

        Use the ``sub`` parameter as the subject to delegate on behalf of
        that user.

        For example::

          >>> account_sub = 'foo@email.com'
          >>> delegate_creds = creds.create_delegated(account_sub)

        Args:
            sub: string, An email address that this service account will
                 act on behalf of (via domain-wide delegation).

        Returns:
            ServiceAccountCredentials, a copy of the current service account
            updated to act on behalf of ``sub``.
        """
    return self.create_with_claims({'sub': sub})