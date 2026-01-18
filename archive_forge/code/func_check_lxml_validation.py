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
def check_lxml_validation(self):
    try:
        schema = lxml_etree.XMLSchema(self.lxml_schema.getroot())
    except lxml_etree.XMLSchemaParseError:
        print('\nSkip lxml.etree.XMLSchema validation test for {!r} ({})'.format(xml_file, TestValidator.__name__))
    else:
        xml_tree = lxml_etree.parse(xml_file)
        if self.errors:
            self.assertFalse(schema.validate(xml_tree), msg=xml_file)
        else:
            self.assertTrue(schema.validate(xml_tree), msg=xml_file)