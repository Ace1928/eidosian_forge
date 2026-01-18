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
def make_schema_test_class(test_file, test_args, test_num, schema_class, check_with_lxml):
    """
    Creates a schema test class.

    :param test_file: the schema test file path.
    :param test_args: line arguments for test case.
    :param test_num: a positive integer number associated with the test case.
    :param schema_class: the schema class to use.
    :param check_with_lxml: if `True` compare with lxml XMLSchema class, reporting anomalies.     Works only for XSD 1.0 tests.
    """
    xsd_file = os.path.relpath(test_file)
    expected_errors = test_args.errors
    expected_warnings = test_args.warnings
    inspect = test_args.inspect
    locations = test_args.locations
    defuse = test_args.defuse
    no_pickle = test_args.no_pickle
    debug_mode = test_args.debug
    codegen = test_args.codegen
    loglevel = logging.DEBUG if debug_mode else None

    class TestSchema(XsdValidatorTestCase):

        @classmethod
        def setUpClass(cls):
            cls.schema_class = schema_class
            cls.errors = []
            cls.longMessage = True
            if debug_mode:
                print('\n##\n## Testing %r schema in debug mode.\n##' % xsd_file)
                pdb.set_trace()

        def check_xsd_file(self):
            if expected_errors > 0:
                schema = schema_class(xsd_file, validation='lax', locations=locations, defuse=defuse, loglevel=loglevel)
            else:
                schema = schema_class(xsd_file, locations=locations, defuse=defuse, loglevel=loglevel)
            self.errors.extend(schema.maps.all_errors)
            if inspect:
                components_ids = set([id(c) for c in schema.maps.iter_components()])
                components_ids.update((id(c) for c in schema.meta_schema.iter_components()))
                missing = [c for c in SchemaObserver.components if id(c) not in components_ids]
                if missing:
                    raise ValueError('schema missing %d components: %r' % (len(missing), missing))
            if not inspect and (not no_pickle):
                try:
                    obj = pickle.dumps(schema)
                    deserialized_schema = pickle.loads(obj)
                except pickle.PicklingError:
                    for e in schema.maps.iter_components():
                        elem = getattr(e, 'elem', getattr(e, 'root', None))
                        if isinstance(elem, PyElementTree.Element):
                            break
                    else:
                        raise
                else:
                    self.assertTrue(isinstance(deserialized_schema, XMLSchemaBase), msg=xsd_file)
                    self.assertEqual(schema.built, deserialized_schema.built, msg=xsd_file)
            if not inspect and (not self.errors):
                xpath_root = schema.xpath_node
                element_nodes = [x for x in xpath_root.iter() if hasattr(x, 'elem')]
                descendants = [x for x in xpath_root.iter_descendants('descendant-or-self')]
                self.assertTrue((x in descendants for x in element_nodes))
                context_xsd_elements = [e.value for e in element_nodes]
                for xsd_element in schema.iter():
                    self.assertIn(xsd_element, context_xsd_elements, msg=xsd_file)
            for xsd_type in schema.maps.iter_components(xsd_classes=XsdType):
                self.assertIn(xsd_type.content_type_label, {'empty', 'simple', 'element-only', 'mixed'}, msg=xsd_file)
            if not expected_errors and schema_class.XSD_VERSION == '1.0':
                try:
                    XMLSchema11(xsd_file, locations=locations, defuse=defuse, loglevel=loglevel)
                except XMLSchemaParseError as err:
                    if not isinstance(err.validator, Xsd11ComplexType) or 'is simple or has a simple content' not in str(err):
                        raise
                    schema = schema_class(xsd_file, validation='lax', locations=locations, defuse=defuse, loglevel=loglevel)
                    for error in schema.all_errors:
                        if not isinstance(err.validator, Xsd11ComplexType) or 'is simple or has a simple content' not in str(err):
                            raise error
            if codegen and PythonGenerator is not None and (not self.errors) and all(('schemaLocation' in e.attrib for e in schema.root if e.tag == XSD_IMPORT)):
                generator = PythonGenerator(schema)
                with tempfile.TemporaryDirectory() as tempdir:
                    cwd = os.getcwd()
                    try:
                        schema.export(tempdir, save_remote=True)
                        os.chdir(tempdir)
                        generator.render_to_files('bindings.py.jinja')
                        spec = importlib.util.spec_from_file_location(tempdir, 'bindings.py')
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                    finally:
                        os.chdir(cwd)

        def check_xsd_file_with_lxml(self, xmlschema_time):
            start_time = time.time()
            lxs = lxml_etree.parse(xsd_file)
            try:
                lxml_etree.XMLSchema(lxs.getroot())
            except lxml_etree.XMLSchemaParseError as err:
                if not self.errors:
                    print('\nSchema error with lxml.etree.XMLSchema for file {!r} ({}): {}'.format(xsd_file, self.__class__.__name__, str(err)))
            else:
                if self.errors:
                    msg = '\nUnrecognized errors with lxml.etree.XMLSchema for file {!r} ({}): {}'
                    print(msg.format(xsd_file, self.__class__.__name__, '\n++++++\n'.join([str(e) for e in self.errors])))
                lxml_schema_time = time.time() - start_time
                if lxml_schema_time >= xmlschema_time:
                    msg = '\nSlower lxml.etree.XMLSchema ({:.3f}s VS {:.3f}s) with file {!r} ({})'
                    print(msg.format(lxml_schema_time, xmlschema_time, xsd_file, self.__class__.__name__))

        def test_xsd_file(self):
            if inspect:
                SchemaObserver.clear()
            del self.errors[:]
            start_time = time.time()
            if expected_warnings > 0:
                with warnings.catch_warnings(record=True) as include_import_warnings:
                    warnings.simplefilter('always')
                    self.check_xsd_file()
                    self.assertEqual(len(include_import_warnings), expected_warnings, msg=xsd_file)
            else:
                self.check_xsd_file()
            if check_with_lxml and lxml_etree is not None:
                self.check_xsd_file_with_lxml(xmlschema_time=time.time() - start_time)
            self.check_errors(xsd_file, expected_errors)
    TestSchema.__name__ = TestSchema.__qualname__ = str('TestSchema{0:03}'.format(test_num))
    return TestSchema