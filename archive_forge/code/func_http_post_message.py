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
def http_post_message(message, relay_state='', typ='SAMLRequest', **kwargs):
    """

    :param message: The message
    :param relay_state: for preserving and conveying state information
    :return: A tuple containing header information and a HTML message.
    """
    if not isinstance(message, str):
        message = str(message)
    if not isinstance(message, bytes):
        message = message.encode('utf-8')
    if typ == 'SAMLRequest' or typ == 'SAMLResponse':
        _msg = base64.b64encode(message)
    else:
        _msg = message
    _msg = _msg.decode('ascii')
    part = {typ: _msg}
    if relay_state:
        part['RelayState'] = relay_state
    return {'headers': [('Content-type', 'application/x-www-form-urlencoded')], 'data': urlencode(part), 'status': 200}