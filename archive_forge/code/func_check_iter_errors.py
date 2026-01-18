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
def check_iter_errors(self):

    def compare_error_reasons(reason, other_reason):
        if ' at 0x' in reason:
            self.assertEqual(OBJ_ID_PATTERN.sub(' at 0xff', reason), OBJ_ID_PATTERN.sub(' at 0xff', other_reason), msg=xml_file)
        else:
            self.assertEqual(reason, other_reason, msg=xml_file)
    errors = list(self.schema.iter_errors(xml_file))
    for e in errors:
        self.assertIsInstance(e.reason, str, msg=xml_file)
    self.assertEqual(len(errors), expected_errors, msg=xml_file)
    module_api_errors = list(xmlschema.iter_errors(xml_file, schema=self.schema))
    self.assertEqual(len(errors), len(module_api_errors), msg=xml_file)
    for e, api_error in zip(errors, module_api_errors):
        compare_error_reasons(e.reason, api_error.reason)
    lazy_errors = list(xmlschema.iter_errors(xml_file, schema=self.schema, lazy=True))
    self.assertEqual(len(errors), len(lazy_errors), msg=xml_file)
    for e, lazy_error in zip(errors, lazy_errors):
        compare_error_reasons(e.reason, lazy_error.reason)