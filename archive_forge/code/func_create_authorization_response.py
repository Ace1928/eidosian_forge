from __future__ import absolute_import, unicode_literals
import base64
import hashlib
import json
import logging
from oauthlib import common
from .. import errors
from .base import GrantTypeBase
def create_authorization_response(self, request, token_handler):
    """
        The client constructs the request URI by adding the following
        parameters to the query component of the authorization endpoint URI
        using the "application/x-www-form-urlencoded" format, per `Appendix B`_:

        response_type
                REQUIRED.  Value MUST be set to "code" for standard OAuth2
                authorization flow.  For OpenID Connect it must be one of
                "code token", "code id_token", or "code token id_token" - we
                essentially test that "code" appears in the response_type.
        client_id
                REQUIRED.  The client identifier as described in `Section 2.2`_.
        redirect_uri
                OPTIONAL.  As described in `Section 3.1.2`_.
        scope
                OPTIONAL.  The scope of the access request as described by
                `Section 3.3`_.
        state
                RECOMMENDED.  An opaque value used by the client to maintain
                state between the request and callback.  The authorization
                server includes this value when redirecting the user-agent back
                to the client.  The parameter SHOULD be used for preventing
                cross-site request forgery as described in `Section 10.12`_.

        The client directs the resource owner to the constructed URI using an
        HTTP redirection response, or by other means available to it via the
        user-agent.

        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :param token_handler: A token handler instance, for example of type
                              oauthlib.oauth2.BearerToken.
        :returns: headers, body, status
        :raises: FatalClientError on invalid redirect URI or client id.

        A few examples::

            >>> from your_validator import your_validator
            >>> request =
            Request('https://example.com/authorize?client_id=valid'
            ...                   '&redirect_uri=http%3A%2F%2Fclient.com%2F')
            >>> from oauthlib.common import Request
            >>> from oauthlib.oauth2 import AuthorizationCodeGrant, BearerToken
            >>> token = BearerToken(your_validator)
            >>> grant = AuthorizationCodeGrant(your_validator)
            >>> request.scopes = ['authorized', 'in', 'some', 'form']
            >>> grant.create_authorization_response(request, token)
            (u'http://client.com/?error=invalid_request&error_description=Missing+response_type+parameter.',
            None, None, 400)
            >>> request =
            Request('https://example.com/authorize?client_id=valid'
            ...                   '&redirect_uri=http%3A%2F%2Fclient.com%2F'
            ...                   '&response_type=code')
            >>> request.scopes = ['authorized', 'in', 'some', 'form']
            >>> grant.create_authorization_response(request, token)
            (u'http://client.com/?code=u3F05aEObJuP2k7DordviIgW5wl52N', None,
            None, 200)
            >>> # If the client id or redirect uri fails validation
            >>> grant.create_authorization_response(request, token)
            Traceback (most recent call last):
                File "<stdin>", line 1, in <module>
                File "oauthlib/oauth2/rfc6749/grant_types.py", line 515, in
                create_authorization_response
                    >>> grant.create_authorization_response(request, token)
                File "oauthlib/oauth2/rfc6749/grant_types.py", line 591, in
                validate_authorization_request
            oauthlib.oauth2.rfc6749.errors.InvalidClientIdError

        .. _`Appendix B`: https://tools.ietf.org/html/rfc6749#appendix-B
        .. _`Section 2.2`: https://tools.ietf.org/html/rfc6749#section-2.2
        .. _`Section 3.1.2`: https://tools.ietf.org/html/rfc6749#section-3.1.2
        .. _`Section 3.3`: https://tools.ietf.org/html/rfc6749#section-3.3
        .. _`Section 10.12`: https://tools.ietf.org/html/rfc6749#section-10.12
        """
    try:
        self.validate_authorization_request(request)
        log.debug('Pre resource owner authorization validation ok for %r.', request)
    except errors.FatalClientError as e:
        log.debug('Fatal client error during validation of %r. %r.', request, e)
        raise
    except errors.OAuth2Error as e:
        log.debug('Client error during validation of %r. %r.', request, e)
        request.redirect_uri = request.redirect_uri or self.error_uri
        redirect_uri = common.add_params_to_uri(request.redirect_uri, e.twotuples, fragment=request.response_mode == 'fragment')
        return ({'Location': redirect_uri}, None, 302)
    grant = self.create_authorization_code(request)
    for modifier in self._code_modifiers:
        grant = modifier(grant, token_handler, request)
    log.debug('Saving grant %r for %r.', grant, request)
    self.request_validator.save_authorization_code(request.client_id, grant, request)
    return self.prepare_authorization_response(request, grant, {}, None, 302)