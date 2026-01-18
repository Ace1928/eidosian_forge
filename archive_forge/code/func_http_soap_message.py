import base64
import logging
from urllib.parse import urlencode
from urllib.parse import urlparse
from xml.etree import ElementTree as ElementTree
import defusedxml.ElementTree
import saml2
from saml2.s_utils import deflate_and_base64_encode
from saml2.sigver import REQ_ORDER
from saml2.sigver import RESP_ORDER
from saml2.xmldsig import SIG_ALLOWED_ALG
def http_soap_message(message):
    return {'headers': [('Content-type', 'application/soap+xml')], 'data': make_soap_enveloped_saml_thingy(message), 'status': 200}