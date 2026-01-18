from __future__ import unicode_literals
import sys
import datetime
from decimal import Decimal
import os
import logging
import hashlib
import warnings
from . import __author__, __copyright__, __license__, __version__
def preprocess_schema(schema, imported_schemas, elements, xsd_uri, dialect, http, cache, force_download, wsdl_basedir, global_namespaces=None, qualified=False):
    """Find schema elements and complex types"""
    from .simplexml import SimpleXMLElement
    local_namespaces = {}
    for k, v in schema[:]:
        if k.startswith('xmlns'):
            local_namespaces[get_local_name(k)] = v
        if k == 'targetNamespace':
            if v == 'urn:DefaultNamespace':
                v = global_namespaces[None]
            local_namespaces[None] = v
        if k == 'elementFormDefault':
            qualified = v == 'qualified'
    for ns in local_namespaces.values():
        if ns not in global_namespaces:
            global_namespaces[ns] = 'ns%s' % len(global_namespaces)
    for element in schema.children() or []:
        if element.get_local_name() in ('import', 'include'):
            schema_namespace = element['namespace']
            schema_location = element['schemaLocation']
            if schema_location is None:
                log.debug('Schema location not provided for %s!' % schema_namespace)
                continue
            if schema_location in imported_schemas:
                log.debug('Schema %s already imported!' % schema_location)
                continue
            imported_schemas[schema_location] = schema_namespace
            log.debug('Importing schema %s from %s' % (schema_namespace, schema_location))
            xml = fetch(schema_location, http, cache, force_download, wsdl_basedir)
            path = os.path.normpath(os.path.join(wsdl_basedir, schema_location))
            path = os.path.dirname(path)
            imported_schema = SimpleXMLElement(xml, namespace=xsd_uri)
            preprocess_schema(imported_schema, imported_schemas, elements, xsd_uri, dialect, http, cache, force_download, path, global_namespaces, qualified)
        element_type = element.get_local_name()
        if element_type in ('element', 'complexType', 'simpleType'):
            namespace = local_namespaces[None]
            element_ns = global_namespaces[ns]
            element_name = element['name']
            log.debug('Parsing Element %s: %s' % (element_type, element_name))
            if element.get_local_name() == 'complexType':
                children = element.children()
            elif element.get_local_name() == 'simpleType':
                children = element('restriction', ns=xsd_uri, error=False)
                if not children:
                    children = element.children()
            elif element.get_local_name() == 'element' and element['type']:
                children = element
            else:
                children = element.children()
                if children:
                    children = children.children()
                elif element.get_local_name() == 'element':
                    children = element
            if children:
                process_element(elements, element_name, children, element_type, xsd_uri, dialect, namespace, qualified)