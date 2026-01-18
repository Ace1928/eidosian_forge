import logging
import re
from xml.etree import ElementTree as ElementTree
import defusedxml.ElementTree
from saml2 import create_class_from_element_tree
from saml2.samlp import NAMESPACE as SAMLP_NAMESPACE
from saml2.schema import soapenv
def make_soap_enveloped_saml_thingy(thingy, headers=None):
    """Returns a soap envelope containing a SAML request
    as a text string.

    :param thingy: The SAML thingy
    :return: The SOAP envelope as a string
    """
    soap_envelope = soapenv.Envelope()
    if headers:
        _header = soapenv.Header()
        _header.add_extension_elements(headers)
        soap_envelope.header = _header
    soap_envelope.body = soapenv.Body()
    soap_envelope.body.add_extension_element(thingy)
    return f'{soap_envelope}'