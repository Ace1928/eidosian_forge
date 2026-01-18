import logging
from saml2 import BINDING_PAOS
from saml2 import BINDING_SOAP
from saml2 import element_to_extension_element
from saml2 import saml
from saml2 import samlp
from saml2 import soap
from saml2.client_base import ACTOR
from saml2.client_base import MIME_PAOS
from saml2.ecp_client import SERVICE
from saml2.profile import ecp
from saml2.profile import paos
from saml2.response import authn_response
from saml2.schema import soapenv
from saml2.server import Server
def ecp_auth_request(cls, entityid=None, relay_state='', sign=None, sign_alg=None, digest_alg=None):
    """Makes an authentication request.

    :param entityid: The entity ID of the IdP to send the request to
    :param relay_state: To where the user should be returned after
        successfull log in.
    :param sign: Whether the request should be signed or not.
    :return: AuthnRequest response
    """
    eelist = []
    my_url = cls.service_urls(BINDING_PAOS)[0]
    paos_request = paos.Request(must_understand='1', actor=ACTOR, response_consumer_url=my_url, service=SERVICE)
    eelist.append(element_to_extension_element(paos_request))
    logger.info(f'entityid: {entityid}, binding: {BINDING_SOAP}')
    location = cls._sso_location(entityid, binding=BINDING_SOAP)
    req_id, authn_req = cls.create_authn_request(location, binding=BINDING_PAOS, service_url_binding=BINDING_PAOS, sign=sign, sign_alg=sign_alg, digest_alg=digest_alg)
    body = soapenv.Body()
    body.extension_elements = [element_to_extension_element(authn_req)]
    idp_list = None
    ecp_request = ecp.Request(actor=ACTOR, must_understand='1', provider_name=None, issuer=saml.Issuer(text=authn_req.issuer.text), idp_list=idp_list)
    eelist.append(element_to_extension_element(ecp_request))
    relay_state = ecp.RelayState(actor=ACTOR, must_understand='1', text=relay_state)
    eelist.append(element_to_extension_element(relay_state))
    header = soapenv.Header()
    header.extension_elements = eelist
    soap_envelope = soapenv.Envelope(header=header, body=body)
    return (req_id, str(soap_envelope))