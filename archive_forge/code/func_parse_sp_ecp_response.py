from http import cookiejar as cookielib
import logging
from saml2 import BINDING_SOAP
from saml2 import SAMLError
from saml2 import saml
from saml2 import samlp
from saml2 import soap
from saml2.client_base import MIME_PAOS
from saml2.config import Config
from saml2.entity import Entity
from saml2.httpbase import dict2set_list
from saml2.httpbase import set_list2dict
from saml2.mdstore import MetadataStore
from saml2.profile import ecp
from saml2.profile import paos
from saml2.s_utils import BadRequest
@staticmethod
def parse_sp_ecp_response(respdict):
    if respdict is None:
        raise SAMLError('Unexpected reply from the SP')
    logger.debug('[P1] SP response dict: %s', respdict)
    authn_request = respdict['body']
    expected_tag = 'AuthnRequest'
    if authn_request.c_tag != expected_tag:
        raise ValueError("Invalid AuthnRequest tag '{invalid}' should be '{valid}'".format(invalid=authn_request.c_tag, valid=expected_tag))
    _relay_state = None
    _paos_request = None
    for item in respdict['header']:
        if item.c_tag == 'RelayState' and item.c_namespace == ecp.NAMESPACE:
            _relay_state = item
        if item.c_tag == 'Request' and item.c_namespace == paos.NAMESPACE:
            _paos_request = item
    if _paos_request is None:
        raise BadRequest('Missing request')
    _rc_url = _paos_request.response_consumer_url
    return {'authn_request': authn_request, 'rc_url': _rc_url, 'relay_state': _relay_state}