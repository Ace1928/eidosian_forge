from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible_collections.community.dns.plugins.module_utils.http import (
def decode_wsdl(node, root_ns, ids):
    href = node.get('href')
    nil = node.get(lxml.etree.QName(_NAMESPACE_XSI, 'nil'))
    id = node.get('id')
    if href is not None:
        if not href.startswith('#'):
            raise WSDLCodingException('Global reference "{0}" not supported!'.format(href))
        href = href[1:]
        if href not in ids:
            raise WSDLCodingException('ID "{0}" not yet defined!'.format(href))
        result = ids[href]
    elif nil == 'true':
        result = None
    else:
        type_with_ns = node.get(lxml.etree.QName(_NAMESPACE_XSI, 'type'))
        if type_with_ns is None:
            raise WSDLCodingException('Element "{0}" has no "xsi:type" tag!'.format(node))
        type, ns = _split_text_namespace(node, type_with_ns)
        if ns is None:
            raise WSDLCodingException('Cannot find namespace for "{0}"!'.format(type_with_ns))
        if ns == _NAMESPACE_XSD:
            if type == 'boolean':
                if node.text == 'true':
                    result = True
                elif node.text == 'false':
                    result = False
                else:
                    raise WSDLCodingException('Invalid value for boolean: "{0}"'.format(node.text))
            elif type == 'int':
                result = int(node.text)
            elif type == 'string':
                result = node.text
            else:
                raise WSDLCodingException('Unknown XSD type "{0}"!'.format(type))
        elif ns == _NAMESPACE_XML_SOAP:
            if type == 'Map':
                result = dict()
                if id is not None:
                    ids[id] = result
                for item in node:
                    if item.tag != 'item':
                        raise WSDLCodingException('Invalid child tag "{0}" in map!'.format(item.tag))
                    key = item.find('key')
                    if key is None:
                        raise WSDLCodingException('Cannot find key for "{0}"!'.format(item))
                    key = decode_wsdl(key, root_ns, ids)
                    value = item.find('value')
                    if value is None:
                        raise WSDLCodingException('Cannot find value for "{0}"!'.format(item))
                    value = decode_wsdl(value, root_ns, ids)
                    result[key] = value
                return result
            else:
                raise WSDLCodingException('Unknown XSD type "{0}"!'.format(type))
        elif ns == _NAMESPACE_XML_SOAP_ENCODING:
            if type == 'Array':
                result = []
                if id is not None:
                    ids[id] = result
                _decode_wsdl_array(result, node, root_ns, ids)
            else:
                raise WSDLCodingException('Unknown XSD type "{0}"!'.format(type))
        elif ns == root_ns:
            array_type = node.get(lxml.etree.QName(_NAMESPACE_XML_SOAP_ENCODING, 'arrayType'))
            if array_type is not None:
                result = []
                if id is not None:
                    ids[id] = result
                _decode_wsdl_array(result, node, root_ns, ids)
            else:
                result = dict()
                if id is not None:
                    ids[id] = result
                for item in node:
                    result[item.tag] = decode_wsdl(item, root_ns, ids)
        else:
            raise WSDLCodingException('Unknown type namespace "{0}" (with type "{1}")!'.format(ns, type))
    if id is not None:
        ids[id] = result
    return result