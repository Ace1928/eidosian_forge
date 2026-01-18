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
def check_json_serialization(self, root, converter=None, **kwargs):
    lossy = converter in (ParkerConverter, AbderaConverter, ColumnarConverter)
    unordered = converter not in (AbderaConverter, JsonMLConverter) or kwargs.get('unordered', False)
    kwargs['decimal_type'] = str
    json_data1 = xmlschema.to_json(root, schema=self.schema, converter=converter, **kwargs)
    if isinstance(json_data1, tuple):
        json_data1 = json_data1[0]
    elem1 = xmlschema.from_json(json_data1, schema=self.schema, path=root.tag, converter=converter, **kwargs)
    if isinstance(elem1, tuple):
        elem1 = elem1[0]
    if lax_encode:
        kwargs['validation'] = kwargs.get('validation', 'lax')
    json_data2 = xmlschema.to_json(elem1, schema=self.schema, converter=converter, **kwargs)
    if isinstance(json_data2, tuple):
        json_data2 = json_data2[0]
    if json_data2 != json_data1 and (lax_encode or lossy or unordered):
        return
    self.assertEqual(json_data2, json_data1, msg=xml_file)