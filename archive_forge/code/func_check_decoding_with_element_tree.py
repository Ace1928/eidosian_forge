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
def check_decoding_with_element_tree(self):
    del self.errors[:]
    del self.chunks[:]

    def do_decoding():
        for obj in self.schema.iter_decode(xml_file):
            if isinstance(obj, (xmlschema.XMLSchemaDecodeError, xmlschema.XMLSchemaValidationError)):
                self.errors.append(obj)
            else:
                self.chunks.append(obj)
    if expected_warnings == 0:
        do_decoding()
    else:
        with warnings.catch_warnings(record=True) as include_import_warnings:
            warnings.simplefilter('always')
            do_decoding()
            self.assertEqual(len(include_import_warnings), expected_warnings, msg=xml_file)
    self.check_errors(xml_file, expected_errors)
    if not self.chunks:
        raise ValueError('No decoded object returned!!')
    elif len(self.chunks) > 1:
        raise ValueError('Too many ({}) decoded objects returned: {}'.format(len(self.chunks), self.chunks))
    elif not self.errors:
        try:
            skip_decoded_data = self.schema.decode(xml_file, validation='skip')
            self.assertEqual(skip_decoded_data, self.chunks[0], msg=xml_file)
        except AssertionError:
            if not lax_encode:
                raise