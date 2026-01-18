from __future__ import absolute_import, unicode_literals
import logging
from oauthlib.common import Request
from oauthlib.oauth2.rfc6749 import utils
from .base import BaseEndpoint, catch_errors_and_unavailability
@property
def response_types(self):
    return self._response_types