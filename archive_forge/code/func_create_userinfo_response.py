import json
import logging
from oauthlib.common import Request
from oauthlib.oauth2.rfc6749 import errors
from oauthlib.oauth2.rfc6749.endpoints.base import (
from oauthlib.oauth2.rfc6749.tokens import BearerToken
@catch_errors_and_unavailability
def create_userinfo_response(self, uri, http_method='GET', body=None, headers=None):
    """Validate BearerToken and return userinfo from RequestValidator

        The UserInfo Endpoint MUST return a
        content-type header to indicate which format is being returned. The
        content-type of the HTTP response MUST be application/json if the
        response body is a text JSON object; the response body SHOULD be encoded
        using UTF-8.
        """
    request = Request(uri, http_method, body, headers)
    request.scopes = ['openid']
    self.validate_userinfo_request(request)
    claims = self.request_validator.get_userinfo_claims(request)
    if claims is None:
        log.error('Userinfo MUST have claims for %r.', request)
        raise errors.ServerError(status_code=500)
    if isinstance(claims, dict):
        resp_headers = {'Content-Type': 'application/json'}
        if 'sub' not in claims:
            log.error('Userinfo MUST have "sub" for %r.', request)
            raise errors.ServerError(status_code=500)
        body = json.dumps(claims)
    elif isinstance(claims, str):
        resp_headers = {'Content-Type': 'application/jwt'}
        body = claims
    else:
        log.error('Userinfo return unknown response for %r.', request)
        raise errors.ServerError(status_code=500)
    log.debug('Userinfo access valid for %r.', request)
    return (resp_headers, body, 200)