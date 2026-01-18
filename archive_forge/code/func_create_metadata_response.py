from __future__ import absolute_import, unicode_literals
import copy
import json
import logging
from ....common import unicode_type
from .base import BaseEndpoint, catch_errors_and_unavailability
from .authorization import AuthorizationEndpoint
from .introspect import IntrospectEndpoint
from .token import TokenEndpoint
from .revocation import RevocationEndpoint
from .. import grant_types
@catch_errors_and_unavailability
def create_metadata_response(self, uri, http_method='GET', body=None, headers=None):
    """Create metadata response"""
    headers = {'Content-Type': 'application/json'}
    return (headers, json.dumps(self.claims), 200)