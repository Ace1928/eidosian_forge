import unittest
import re
import os
from textwrap import dedent
from xml.etree.ElementTree import Element, iselement
from xmlschema.exceptions import XMLSchemaValueError
from xmlschema.names import XSD_NAMESPACE, XSI_NAMESPACE, XSD_SCHEMA
from xmlschema.helpers import get_namespace
from xmlschema.resources import fetch_namespaces
from xmlschema.validators import XMLSchema10
from ._helpers import etree_elements_assert_equal
def get_schema_source(self, source):
    """
        Returns a schema source that can be used to create an XMLSchema instance.

        :param source: A string or an ElementTree's Element.
        :return: An schema source string, an ElementTree's Element or a full pathname.
        """
    if iselement(source):
        if source.tag in (XSD_SCHEMA, 'schema'):
            return source
        elif get_namespace(source.tag):
            raise XMLSchemaValueError('source %r namespace has to be empty.' % source)
        elif source.tag not in {'element', 'attribute', 'simpleType', 'complexType', 'group', 'attributeGroup', 'notation'}:
            raise XMLSchemaValueError('% is not an XSD global definition/declaration.' % source)
        root = Element('schema', attrib={'xmlns:xs': XSD_NAMESPACE, 'xmlns:xsi': XSI_NAMESPACE, 'elementFormDefault': 'qualified', 'version': self.schema_class.XSD_VERSION})
        root.append(source)
        return root
    else:
        source = dedent(source.strip())
        if not source.startswith('<'):
            return self.casepath(source)
        elif source.startswith('<?xml ') or source.startswith('<xs:schema '):
            return source
        else:
            return SCHEMA_TEMPLATE.format(self.schema_class.XSD_VERSION, source)