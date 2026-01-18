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
def ecp_capable(headers):
    if MIME_PAOS in headers['Accept']:
        if 'PAOS' in headers:
            if f'ver="{paos.NAMESPACE}";"{SERVICE}"' in headers['PAOS']:
                return True
    return False