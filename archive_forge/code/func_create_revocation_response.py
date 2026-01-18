from __future__ import absolute_import, unicode_literals
import logging
from oauthlib.common import Request
from ..errors import OAuth2Error, UnsupportedTokenTypeError
from .base import BaseEndpoint, catch_errors_and_unavailability
@catch_errors_and_unavailability
def create_revocation_response(self, uri, http_method='POST', body=None, headers=None):
    """Revoke supplied access or refresh token.


        The authorization server responds with HTTP status code 200 if the
        token has been revoked sucessfully or if the client submitted an
        invalid token.

        Note: invalid tokens do not cause an error response since the client
        cannot handle such an error in a reasonable way.  Moreover, the purpose
        of the revocation request, invalidating the particular token, is
        already achieved.

        The content of the response body is ignored by the client as all
        necessary information is conveyed in the response code.

        An invalid token type hint value is ignored by the authorization server
        and does not influence the revocation response.
        """
    resp_headers = {'Content-Type': 'application/json', 'Cache-Control': 'no-store', 'Pragma': 'no-cache'}
    request = Request(uri, http_method=http_method, body=body, headers=headers)
    try:
        self.validate_revocation_request(request)
        log.debug('Token revocation valid for %r.', request)
    except OAuth2Error as e:
        log.debug('Client error during validation of %r. %r.', request, e)
        response_body = e.json
        if self.enable_jsonp and request.callback:
            response_body = '%s(%s);' % (request.callback, response_body)
        resp_headers.update(e.headers)
        return (resp_headers, response_body, e.status_code)
    self.request_validator.revoke_token(request.token, request.token_type_hint, request)
    response_body = ''
    if self.enable_jsonp and request.callback:
        response_body = request.callback + '();'
    return ({}, response_body, 200)