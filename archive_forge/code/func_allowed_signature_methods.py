from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
@property
def allowed_signature_methods(self):
    return SIGNATURE_METHODS