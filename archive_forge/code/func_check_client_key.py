from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
def check_client_key(self, client_key):
    """Check that the client key only contains safe characters

        and is no shorter than lower and no longer than upper.
        """
    lower, upper = self.client_key_length
    return set(client_key) <= self.safe_characters and lower <= len(client_key) <= upper