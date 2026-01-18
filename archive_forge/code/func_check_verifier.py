from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
def check_verifier(self, verifier):
    """Checks that the verifier contains only safe characters

        and is no shorter than lower and no longer than upper.
        """
    lower, upper = self.verifier_length
    return set(verifier) <= self.safe_characters and lower <= len(verifier) <= upper