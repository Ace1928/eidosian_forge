from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible_collections.community.dns.plugins.module_utils.http import (
def encode_wsdl(node, value):
    if value is None:
        node.set(lxml.etree.QName(_NAMESPACE_XSI, 'nil').text, 'true')
    elif isinstance(value, string_types):
        _set_type(node, 'xsd:string')
        node.text = value
    elif isinstance(value, int):
        _set_type(node, 'xsd:int')
        node.text = str(value)
    elif isinstance(value, bool):
        _set_type(node, 'xsd:boolean')
        node.text = 'true' if value else 'false'
    elif isinstance(value, dict):
        _set_type(node, 'Map', _NAMESPACE_XML_SOAP)
        for key, val in sorted(value.items()):
            child = lxml.etree.Element('item')
            ke = lxml.etree.Element('key')
            encode_wsdl(ke, key)
            child.append(ke)
            ve = lxml.etree.Element('value')
            encode_wsdl(ve, val)
            child.append(ve)
            node.append(child)
    elif isinstance(value, list):
        _set_type(node, 'SOAP-ENC:Array')
        for elt in value:
            child = lxml.etree.Element('item')
            encode_wsdl(child, elt)
            node.append(child)
    else:
        raise WSDLCodingException('Do not know how to encode {0}!'.format(type(value)))