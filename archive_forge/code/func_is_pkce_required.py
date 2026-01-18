from __future__ import absolute_import, unicode_literals
import logging
def is_pkce_required(self, client_id, request):
    """Determine if current request requires PKCE.

    Default, False.
        This is called for both "authorization" and "token" requests.

        Override this method by ``return True`` to enable PKCE for everyone.
        You might want to enable it only for public clients.
        Note that PKCE can also be used in addition of a client authentication.

        OAuth 2.0 public clients utilizing the Authorization Code Grant are
        susceptible to the authorization code interception attack.  This
        specification describes the attack as well as a technique to mitigate
        against the threat through the use of Proof Key for Code Exchange
        (PKCE, pronounced "pixy"). See `RFC7636`_.

        :param client_id: Client identifier.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :rtype: True or False

        Method is used by:
            - Authorization Code Grant

        .. _`RFC7636`: https://tools.ietf.org/html/rfc7636
        """
    return False