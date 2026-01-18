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
def check_validation_with_generated_code(self):
    generator = PythonGenerator(self.schema)
    python_module = generator.render('bindings.py.jinja')[0]
    ast_module = ast.parse(python_module)
    self.assertIsInstance(ast_module, ast.Module)
    with tempfile.TemporaryDirectory() as tempdir:
        module_name = '{}.py'.format(self.schema.name.rstrip('.xsd'))
        cwd = os.getcwd()
        try:
            self.schema.export(tempdir, save_remote=True)
            os.chdir(tempdir)
            with open(module_name, 'w') as fp:
                fp.write(python_module)
            spec = importlib.util.spec_from_file_location(tempdir, module_name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            xml_root = ElementTree.parse(os.path.join(cwd, xml_file)).getroot()
            bindings = [x for x in filter(lambda x: x.endswith('Binding'), dir(module))]
            if len(bindings) == 1:
                class_name = bindings[0]
            else:
                class_name = '{}Binding'.format(local_name(xml_root.tag).title().replace('_', ''))
            binding_class = getattr(module, class_name)
            xml_data = binding_class.fromsource(os.path.join(cwd, xml_file))
            self.assertEqual(xml_data.tag, xml_root.tag)
        finally:
            os.chdir(cwd)