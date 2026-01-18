from __future__ import absolute_import, unicode_literals
import logging
def get_id_token(self, token, token_handler, request):
    """Get OpenID Connect ID token

        In the OpenID Connect workflows when an ID Token is requested this
        method is called.
        Subclasses should implement the construction, signing and optional
        encryption of the
        ID Token as described in the OpenID Connect spec.

        In addition to the standard OAuth2 request properties, the request may
        also contain
        these OIDC specific properties which are useful to this method:

            - nonce, if workflow is implicit or hybrid and it was provided
            - claims, if provided to the original Authorization Code request

        The token parameter is a dict which may contain an ``access_token``
        entry, in which
        case the resulting ID Token *should* include a calculated ``at_hash``
        claim.

        Similarly, when the request parameter has a ``code`` property defined,
        the ID Token
        *should* include a calculated ``c_hash`` claim.

        http://openid.net/specs/openid-connect-core-1_0.html (sections
        `3.1.3.6`_, `3.2.2.10`_, `3.3.2.11`_)

        .. _`3.1.3.6`:
        http://openid.net/specs/openid-connect-core-1_0.html#CodeIDToken
        .. _`3.2.2.10`:
        http://openid.net/specs/openid-connect-core-1_0.html#ImplicitIDToken
        .. _`3.3.2.11`:
        http://openid.net/specs/openid-connect-core-1_0.html#HybridIDToken

        :param token: A Bearer token dict.
        :param token_handler: The token handler (BearerToken class)
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :return: The ID Token (a JWS signed JWT)
        """
    raise NotImplementedError('Subclasses must implement this method.')