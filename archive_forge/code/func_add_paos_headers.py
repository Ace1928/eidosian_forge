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
def add_paos_headers(headers=None):
    if headers:
        headers = set_list2dict(headers)
        headers['PAOS'] = PAOS_HEADER_INFO
        if 'Accept' in headers:
            headers['Accept'] += f';{MIME_PAOS}'
        elif 'accept' in headers:
            headers['Accept'] = headers['accept']
            headers['Accept'] += f';{MIME_PAOS}'
            del headers['accept']
        headers = dict2set_list(headers)
    else:
        headers = [('Accept', f'text/html; {MIME_PAOS}'), ('PAOS', PAOS_HEADER_INFO)]
    return headers