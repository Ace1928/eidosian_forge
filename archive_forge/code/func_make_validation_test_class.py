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
def make_validation_test_class(test_file, test_args, test_num, schema_class, check_with_lxml):
    """
    Creates a test class for checking xml instance validation.

    :param test_file: the XML test file path.
    :param test_args: line arguments for test case.
    :param test_num: a positive integer number associated with the test case.
    :param schema_class: the schema class to use.
    :param check_with_lxml: if `True` compare with lxml XMLSchema class, reporting anomalies.     Works only for XSD 1.0 tests.
    """
    xml_file = os.path.relpath(test_file)
    msg_tmpl = '%s: {0}:\n\n{1}' % xml_file
    expected_errors = test_args.errors
    expected_warnings = test_args.warnings
    inspect = test_args.inspect
    locations = test_args.locations
    defuse = test_args.defuse
    validation_only = test_args.validation_only
    no_pickle = test_args.no_pickle
    lax_encode = test_args.lax_encode
    debug_mode = test_args.debug
    codegen = test_args.codegen

    class TestValidator(XsdValidatorTestCase):

        @classmethod
        def setUpClass(cls):
            cls.schema_class = schema_class
            source, _locations = xmlschema.fetch_schema_locations(xml_file, locations)
            cls.schema = schema_class(source, validation='lax', locations=_locations, defuse=defuse)
            if check_with_lxml and lxml_etree is not None:
                cls.lxml_schema = lxml_etree.parse(source)
            cls.errors = []
            cls.chunks = []
            cls.longMessage = True
            if debug_mode:
                print('\n##\n## Testing %r validation in debug mode.\n##' % xml_file)
                pdb.set_trace()

        def check_decode_encode(self, root, converter=None, **kwargs):
            namespaces = kwargs.get('namespaces', {})
            lossy = converter in (ParkerConverter, AbderaConverter, ColumnarConverter)
            losslessly = converter is JsonMLConverter
            unordered = converter not in (AbderaConverter, JsonMLConverter) or kwargs.get('unordered', False)
            decoded_data1 = self.schema.decode(root, converter=converter, **kwargs)
            if isinstance(decoded_data1, tuple):
                decoded_data1 = decoded_data1[0]
            for _ in iter_nested_items(decoded_data1):
                pass
            try:
                elem1 = self.schema.encode(decoded_data1, path=root.tag, converter=converter, **kwargs)
            except XMLSchemaValidationError as err:
                raise AssertionError(msg_tmpl.format('error during re-encoding', str(err)))
            if isinstance(elem1, tuple):
                if converter is not ParkerConverter and converter is not ColumnarConverter:
                    for e in elem1[1]:
                        self.check_namespace_prefixes(str(e))
                elem1 = elem1[0]
            if namespaces and all(('ns%d' % k not in namespaces for k in range(10))):
                self.check_namespace_prefixes(etree_tostring(elem1, namespaces=namespaces))
            try:
                etree_elements_assert_equal(root, elem1, strict=False, unordered=unordered)
            except AssertionError as err:
                if lax_encode:
                    pass
                elif lossy or unordered:
                    pass
                elif losslessly:
                    if debug_mode:
                        pdb.set_trace()
                    raise AssertionError(msg_tmpl.format('encoded tree differs from original', str(err)))
                else:
                    decoded_data2 = self.schema.decode(elem1, converter=converter, **kwargs)
                    if isinstance(decoded_data2, tuple):
                        decoded_data2 = decoded_data2[0]
                    try:
                        self.assertEqual(decoded_data1, decoded_data2, msg=xml_file)
                    except AssertionError:
                        if debug_mode:
                            pdb.set_trace()
                        raise
                    elem2 = self.schema.encode(decoded_data2, path=root.tag, converter=converter, **kwargs)
                    if isinstance(elem2, tuple):
                        elem2 = elem2[0]
                    try:
                        etree_elements_assert_equal(elem1, elem2, strict=False, unordered=unordered)
                    except AssertionError as err:
                        if debug_mode:
                            pdb.set_trace()
                        raise AssertionError(msg_tmpl.format('encoded tree differs after second pass', str(err)))

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

        def check_schema_serialization(self):
            serialized_schema = pickle.dumps(self.schema)
            deserialized_schema = pickle.loads(serialized_schema)
            deserialized_errors = []
            deserialized_chunks = []
            for obj in deserialized_schema.iter_decode(xml_file):
                if isinstance(obj, xmlschema.XMLSchemaValidationError):
                    deserialized_errors.append(obj)
                else:
                    deserialized_chunks.append(obj)
            self.assertEqual(len(deserialized_errors), len(self.errors), msg=xml_file)
            self.assertEqual(deserialized_chunks, self.chunks, msg=xml_file)

        def check_decode_api(self):
            strict_decoded_data = self.schema.decode(xml_file)
            lax_decoded_data = self.schema.decode(xml_file, validation='lax')
            skip_decoded_data = self.schema.decode(xml_file, validation='skip')
            self.assertEqual(strict_decoded_data, self.chunks[0], msg=xml_file)
            self.assertEqual(lax_decoded_data[0], self.chunks[0], msg=xml_file)
            self.assertEqual(skip_decoded_data, self.chunks[0], msg=xml_file)

        def check_data_conversion_with_element_tree(self):
            root = ElementTree.parse(xml_file).getroot()
            namespaces = fetch_namespaces(xml_file)
            options = {'namespaces': namespaces}
            self.check_decode_encode(root, cdata_prefix='#', **options)
            self.check_decode_encode(root, UnorderedConverter, cdata_prefix='#', **options)
            self.check_decode_encode(root, ParkerConverter, validation='lax', **options)
            self.check_decode_encode(root, ParkerConverter, validation='skip', **options)
            self.check_decode_encode(root, BadgerFishConverter, **options)
            self.check_decode_encode(root, AbderaConverter, **options)
            self.check_decode_encode(root, JsonMLConverter, **options)
            self.check_decode_encode(root, ColumnarConverter, validation='lax', **options)
            self.check_decode_encode(root, DataElementConverter, **options)
            self.check_decode_encode(root, DataBindingConverter, **options)
            self.schema.maps.clear_bindings()
            self.check_json_serialization(root, cdata_prefix='#', **options)
            self.check_json_serialization(root, UnorderedConverter, **options)
            self.check_json_serialization(root, ParkerConverter, validation='lax', **options)
            self.check_json_serialization(root, ParkerConverter, validation='skip', **options)
            self.check_json_serialization(root, BadgerFishConverter, **options)
            self.check_json_serialization(root, AbderaConverter, **options)
            self.check_json_serialization(root, JsonMLConverter, **options)
            self.check_json_serialization(root, ColumnarConverter, validation='lax', **options)
            self.check_decode_to_objects(root)
            self.check_decode_to_objects(root, with_bindings=True)
            self.schema.maps.clear_bindings()

        def check_decode_to_objects(self, root, with_bindings=False):
            data_element = self.schema.to_objects(xml_file, with_bindings)
            self.assertIsInstance(data_element, DataElement)
            self.assertEqual(data_element.tag, root.tag)
            if not with_bindings:
                self.assertIs(data_element.__class__, DataElement)
            else:
                self.assertEqual(data_element.tag, root.tag)
                self.assertTrue(data_element.__class__.__name__.endswith('Binding'))

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

        def check_validate_and_is_valid_api(self):
            if expected_errors:
                self.assertFalse(self.schema.is_valid(xml_file), msg=xml_file)
                with self.assertRaises(XMLSchemaValidationError, msg=xml_file):
                    self.schema.validate(xml_file)
            else:
                self.assertTrue(self.schema.is_valid(xml_file), msg=xml_file)
                self.assertIsNone(self.schema.validate(xml_file), msg=xml_file)

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

        def test_xml_document_validation(self):
            if not validation_only:
                self.check_decoding_with_element_tree()
                if not inspect and (not no_pickle):
                    self.check_schema_serialization()
                if not self.errors:
                    self.check_data_conversion_with_element_tree()
                if lxml_etree is not None:
                    self.check_data_conversion_with_lxml()
            self.check_iter_errors()
            self.check_validate_and_is_valid_api()
            if check_with_lxml and lxml_etree is not None:
                self.check_lxml_validation()
            if not validation_only and codegen and (PythonGenerator is not None) and (not self.errors) and (not self.schema.all_errors) and all(('schemaLocation' in e.attrib for e in self.schema.root if e.tag == XSD_IMPORT)):
                self.check_validation_with_generated_code()
    TestValidator.__name__ = TestValidator.__qualname__ = 'TestValidator{0:03}'.format(test_num)
    return TestValidator