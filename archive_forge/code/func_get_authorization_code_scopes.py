from __future__ import absolute_import, unicode_literals
import logging
def get_authorization_code_scopes(self, client_id, code, redirect_uri, request):
    """ Extracts scopes from saved authorization code.

        The scopes returned by this method is used to route token requests
        based on scopes passed to Authorization Code requests.

        With that the token endpoint knows when to include OpenIDConnect
        id_token in token response only based on authorization code scopes.

        Only code param should be sufficient to retrieve grant code from
        any storage you are using, `client_id` and `redirect_uri` can gave a
        blank value `""` don't forget to check it before using those values
        in a select query if a database is used.

        :param client_id: Unicode client identifier.
        :param code: Unicode authorization code grant.
        :param redirect_uri: Unicode absolute URI.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :return: A list of scopes

        Method is used by:
            - Authorization Token Grant Dispatcher
        """
    raise NotImplementedError('Subclasses must implement this method.')