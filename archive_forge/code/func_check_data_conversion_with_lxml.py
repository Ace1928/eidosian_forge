import pdb
import os
import ast
import pickle
import re
import time
import logging
import importlib
import tempfile
import warnings
from xml.etree import ElementTree
from elementpath.etree import PyElementTree, etree_tostring
import xmlschema
from xmlschema import XMLSchemaBase, XMLSchema11, XMLSchemaValidationError, \
from xmlschema.names import XSD_IMPORT
from xmlschema.helpers import local_name
from xmlschema.resources import fetch_namespaces
from xmlschema.validators import XsdType, Xsd11ComplexType
from xmlschema.dataobjects import DataElementConverter, DataBindingConverter, DataElement
from ._helpers import iter_nested_items, etree_elements_assert_equal
from ._case_class import XsdValidatorTestCase
from ._observers import SchemaObserver
def check_data_conversion_with_lxml(self):
    xml_tree = lxml_etree.parse(xml_file)
    namespaces = fetch_namespaces(xml_file)
    lxml_errors = []
    lxml_decoded_chunks = []
    for obj in self.schema.iter_decode(xml_tree, namespaces=namespaces):
        if isinstance(obj, xmlschema.XMLSchemaValidationError):
            lxml_errors.append(obj)
        else:
            lxml_decoded_chunks.append(obj)
    self.assertEqual(lxml_decoded_chunks, self.chunks, msg=xml_file)
    self.assertEqual(len(lxml_errors), len(self.errors), msg=xml_file)
    if not lxml_errors:
        root = xml_tree.getroot()
        if namespaces.get(''):
            namespaces['tns0'] = namespaces['']
        options = {'etree_element_class': lxml_etree_element, 'namespaces': namespaces}
        self.check_decode_encode(root, cdata_prefix='#', **options)
        self.check_decode_encode(root, ParkerConverter, validation='lax', **options)
        self.check_decode_encode(root, ParkerConverter, validation='skip', **options)
        self.check_decode_encode(root, BadgerFishConverter, **options)
        self.check_decode_encode(root, AbderaConverter, **options)
        self.check_decode_encode(root, JsonMLConverter, **options)
        self.check_decode_encode(root, UnorderedConverter, cdata_prefix='#', **options)
        self.check_json_serialization(root, cdata_prefix='#', **options)
        self.check_json_serialization(root, ParkerConverter, validation='lax', **options)
        self.check_json_serialization(root, ParkerConverter, validation='skip', **options)
        self.check_json_serialization(root, BadgerFishConverter, **options)
        self.check_json_serialization(root, AbderaConverter, **options)
        self.check_json_serialization(root, JsonMLConverter, **options)
        self.check_json_serialization(root, UnorderedConverter, **options)