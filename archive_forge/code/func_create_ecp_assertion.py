from keystoneclient import base
def create_ecp_assertion(self, service_provider, token_id):
    """Create an ECP wrapped SAML assertion from a token.

        Equivalent Identity API call:
        POST /auth/OS-FEDERATION/saml2/ecp

        :param service_provider: Service Provider resource.
        :type service_provider: string
        :param token_id: Token to transform to SAML assertion.
        :type token_id: string

        :returns: SAML representation of token_id, wrapped in ECP envelope
        :rtype: string
        """
    headers, body = self._create_common_request(service_provider, token_id)
    resp, body = self.client.post(ECP_ENDPOINT, json=body, headers=headers)
    return self._prepare_return_value(resp, resp.text)