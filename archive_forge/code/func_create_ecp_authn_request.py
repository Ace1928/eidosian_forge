import logging
import threading
import time
from typing import Mapping
from urllib.parse import parse_qs
from urllib.parse import urlencode
from urllib.parse import urlparse
from warnings import warn as _warn
import saml2
from saml2 import BINDING_HTTP_POST
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import BINDING_PAOS
from saml2 import BINDING_SOAP
from saml2 import SAMLError
from saml2 import saml
from saml2 import samlp
from saml2 import soap
from saml2.entity import Entity
from saml2.extension import sp_type
from saml2.extension.requested_attributes import RequestedAttribute
from saml2.extension.requested_attributes import RequestedAttributes
from saml2.mdstore import locations
from saml2.population import Population
from saml2.profile import ecp
from saml2.profile import paos
from saml2.response import AssertionIDResponse
from saml2.response import AttributeResponse
from saml2.response import AuthnQueryResponse
from saml2.response import AuthnResponse
from saml2.response import AuthzResponse
from saml2.response import NameIDMappingResponse
from saml2.response import StatusError
from saml2.s_utils import UnravelError
from saml2.s_utils import do_attributes
from saml2.s_utils import signature
from saml2.saml import NAMEID_FORMAT_PERSISTENT
from saml2.saml import NAMEID_FORMAT_TRANSIENT
from saml2.saml import AuthnContextClassRef
from saml2.samlp import AttributeQuery
from saml2.samlp import AuthnQuery
from saml2.samlp import AuthnRequest
from saml2.samlp import AuthzDecisionQuery
from saml2.samlp import Extensions
from saml2.samlp import NameIDMappingRequest
from saml2.samlp import RequestedAuthnContext
from saml2.soap import make_soap_enveloped_saml_thingy
def create_ecp_authn_request(self, entityid=None, relay_state='', sign=None, sign_alg=None, digest_alg=None, **kwargs):
    """Makes an authentication request.

        :param entityid: The entity ID of the IdP to send the request to
        :param relay_state: A token that can be used by the SP to know
            where to continue the conversation with the client
        :param sign: Whether the request should be signed or not.
        :return: SOAP message with the AuthnRequest
        """
    my_url = self.service_urls(BINDING_PAOS)[0]
    paos_request = paos.Request(must_understand='1', actor=ACTOR, response_consumer_url=my_url, service=ECP_SERVICE)
    relay_state = ecp.RelayState(actor=ACTOR, must_understand='1', text=relay_state)
    try:
        authn_req = kwargs['authn_req']
        try:
            req_id = authn_req.id
        except AttributeError:
            req_id = 0
    except KeyError:
        try:
            _binding = kwargs['binding']
        except KeyError:
            _binding = BINDING_SOAP
            kwargs['binding'] = _binding
        logger.debug('entityid: %s, binding: %s', entityid, _binding)
        _, location = self.pick_binding('single_sign_on_service', [_binding], entity_id=entityid)
        req_id, authn_req = self.create_authn_request(location, service_url_binding=BINDING_PAOS, sign=sign, sign_alg=sign_alg, digest_alg=digest_alg, **kwargs)
    soap_envelope = make_soap_enveloped_saml_thingy(authn_req, [paos_request, relay_state])
    return (req_id, str(soap_envelope))