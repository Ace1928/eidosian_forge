from __future__ import absolute_import, unicode_literals
import json
import logging
from oauthlib.common import Request
from ..errors import OAuth2Error, UnsupportedTokenTypeError
from .base import BaseEndpoint, catch_errors_and_unavailability
@catch_errors_and_unavailability
def create_introspect_response(self, uri, http_method='POST', body=None, headers=None):
    """Create introspect valid or invalid response

        If the authorization server is unable to determine the state
        of the token without additional information, it SHOULD return
        an introspection response indicating the token is not active
        as described in Section 2.2.
        """
    resp_headers = {'Content-Type': 'application/json', 'Cache-Control': 'no-store', 'Pragma': 'no-cache'}
    request = Request(uri, http_method, body, headers)
    try:
        self.validate_introspect_request(request)
        log.debug('Token introspect valid for %r.', request)
    except OAuth2Error as e:
        log.debug('Client error during validation of %r. %r.', request, e)
        resp_headers.update(e.headers)
        return (resp_headers, e.json, e.status_code)
    claims = self.request_validator.introspect_token(request.token, request.token_type_hint, request)
    if claims is None:
        return (resp_headers, json.dumps(dict(active=False)), 200)
    if 'active' in claims:
        claims.pop('active')
    return (resp_headers, json.dumps(dict(active=True, **claims)), 200)