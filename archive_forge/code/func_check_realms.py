from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
def check_realms(self, realms):
    """Check that the realm is one of a set allowed realms."""
    return all((r in self.realms for r in realms))