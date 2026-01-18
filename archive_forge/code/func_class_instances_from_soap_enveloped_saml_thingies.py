import logging
import re
from xml.etree import ElementTree as ElementTree
import defusedxml.ElementTree
from saml2 import create_class_from_element_tree
from saml2.samlp import NAMESPACE as SAMLP_NAMESPACE
from saml2.schema import soapenv
def class_instances_from_soap_enveloped_saml_thingies(text, modules):
    """Parses a SOAP enveloped header and body SAML thing and returns the
    thing as a dictionary class instance.

    :param text: The SOAP object as XML
    :param modules: modules representing xsd schemas
    :return: The body and headers as class instances
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
    env = {'header': [], 'body': None}
    for part in envelope:
        if part.tag == '{%s}Body' % soapenv.NAMESPACE:
            if len(envelope) < 1:
                raise Exception('No items in envelope part.')
            env['body'] = instanciate_class(part[0], modules)
        elif part.tag == '{%s}Header' % soapenv.NAMESPACE:
            for item in part:
                env['header'].append(instanciate_class(item, modules))
    return env