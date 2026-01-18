import logging
from oauthlib.oauth2.rfc6749.request_validator import (
def get_userinfo_claims(self, request):
    """Return the UserInfo claims in JSON or Signed or Encrypted.

        The UserInfo Claims MUST be returned as the members of a JSON object
         unless a signed or encrypted response was requested during Client
         Registration. The Claims defined in Section 5.1 can be returned, as can
         additional Claims not specified there.

        For privacy reasons, OpenID Providers MAY elect to not return values for
        some requested Claims.

        If a Claim is not returned, that Claim Name SHOULD be omitted from the
        JSON object representing the Claims; it SHOULD NOT be present with a
        null or empty string value.

        The sub (subject) Claim MUST always be returned in the UserInfo
        Response.

        Upon receipt of the UserInfo Request, the UserInfo Endpoint MUST return
        the JSON Serialization of the UserInfo Response as in Section 13.3 in
        the HTTP response body unless a different format was specified during
        Registration [OpenID.Registration].

        If the UserInfo Response is signed and/or encrypted, then the Claims are
        returned in a JWT and the content-type MUST be application/jwt. The
        response MAY be encrypted without also being signed. If both signing and
        encryption are requested, the response MUST be signed then encrypted,
        with the result being a Nested JWT, as defined in [JWT].

        If signed, the UserInfo Response SHOULD contain the Claims iss (issuer)
        and aud (audience) as members. The iss value SHOULD be the OP's Issuer
        Identifier URL. The aud value SHOULD be or include the RP's Client ID
        value.

        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :rtype: Claims as a dict OR JWT/JWS/JWE as a string

        Method is used by:
            UserInfoEndpoint
        """