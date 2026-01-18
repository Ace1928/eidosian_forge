from __future__ import absolute_import, unicode_literals
import logging
from oauthlib.common import Request
from .base import BaseEndpoint, catch_errors_and_unavailability
@property
def default_token(self):
    return self._default_token