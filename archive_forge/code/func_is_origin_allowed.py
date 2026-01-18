import logging
def is_origin_allowed(self, client_id, origin, request, *args, **kwargs):
    """Indicate if the given origin is allowed to access the token endpoint
        via Cross-Origin Resource Sharing (CORS).  CORS is used by browser-based
        clients, such as Single-Page Applications, to perform the Authorization
        Code Grant.

        (Note:  If performing Authorization Code Grant via a public client such
        as a browser, you should use PKCE as well.)

        If this method returns true, the appropriate CORS headers will be added
        to the response.  By default this method always returns False, meaning
        CORS is disabled.

        :param client_id: Unicode client identifier.
        :param redirect_uri: Unicode origin.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :rtype: bool

        Method is used by:
            - Authorization Code Grant
            - Refresh Token Grant

        """
    return False