import logging
import re
from xml.etree import ElementTree as ElementTree
import defusedxml.ElementTree
from saml2 import create_class_from_element_tree
from saml2.samlp import NAMESPACE as SAMLP_NAMESPACE
from saml2.schema import soapenv
def open_soap_envelope(text):
    """

    :param text: SOAP message
    :return: dictionary with two keys "body"/"header"
    """
    try:
        envelope = defusedxml.ElementTree.fromstring(text)
    except Exception as exc:
        raise XmlParseError(f'{exc}')
    envelope_tag = '{%s}Envelope' % soapenv.NAMESPACE
    if envelope.tag != envelope_tag:
        raise ValueError(f"Invalid envelope tag '{envelope.tag}' should be '{envelope_tag}'")
    if len(envelope) < 1:
        raise Exception('No items in envelope.')
    content = {'header': [], 'body': None}
    for part in envelope:
        if part.tag == '{%s}Body' % soapenv.NAMESPACE:
            if len(envelope) < 1:
                raise Exception('No items in envelope part.')
            content['body'] = ElementTree.tostring(part[0], encoding='UTF-8')
        elif part.tag == '{%s}Header' % soapenv.NAMESPACE:
            for item in part:
                _str = ElementTree.tostring(item, encoding='UTF-8')
                content['header'].append(_str)
    return content