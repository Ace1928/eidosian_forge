import re
import argparse
import os
import fileinput
import logging
from xmlschema.cli import xsd_version_number, defuse_data
from xmlschema.validators import XMLSchema10, XMLSchema11
from ._observers import ObservedXMLSchema10, ObservedXMLSchema11
def factory_tests(test_class_builder, testfiles, suffix, check_with_lxml=False, codegen=False, verbosity=1):
    """
    Factory function for file based schema/validation cases.

    :param test_class_builder: the test class builder function.
    :param testfiles: a single or a list of testfiles indexes.
    :param suffix: the suffix ('xml' or 'xsd') to consider for cases.
    :param check_with_lxml: if `True` compare with lxml XMLSchema class,     reporting anomalies. Works only for XSD 1.0 tests.
    :param codegen: if `True` is provided checks code generation with XML data     bindings module for all tests. For default is `False` and code generation     is tested only for the cases where the same option is provided.
    :param verbosity: the unittest's verbosity, can be 0, 1 or 2.
    :return: a list of test classes.
    """
    test_classes = {}
    test_num = 0
    debug_mode = False
    line_buffer = []
    test_line_parser = get_test_line_args_parser()
    for line in fileinput.input(testfiles):
        line = line.strip()
        if not line or line[0] == '#':
            if not line_buffer:
                continue
            else:
                raise SyntaxError('Empty continuation at line %d!' % fileinput.filelineno())
        elif '#' in line:
            line = line.split('#', 1)[0].rstrip()
        if line[-1] == '\\':
            line_buffer.append(line[:-1].strip())
            continue
        elif line_buffer:
            line_buffer.append(line)
            line = ' '.join(line_buffer)
            del line_buffer[:]
        test_args = test_line_parser.parse_args(get_test_args(line))
        if test_args.locations is not None:
            test_args.locations = {k.strip('\'"'): v for k, v in test_args.locations}
        if codegen:
            test_args.codegen = True
        test_file = os.path.join(os.path.dirname(fileinput.filename()), test_args.filename)
        if os.path.isdir(test_file):
            logger.debug('Skip %s: is a directory.', test_file)
            continue
        elif os.path.splitext(test_file)[1].lower() != '.%s' % suffix:
            logger.debug('Skip %s: wrong suffix.', test_file)
            continue
        elif not os.path.isfile(test_file):
            logger.error('Skip %s: is not a file.', test_file)
            continue
        test_num += 1
        if debug_mode:
            if not test_args.debug:
                continue
        elif test_args.debug:
            debug_mode = True
            msg = 'Debug mode activated: discard previous %r test classes.'
            logger.debug(msg, len(test_classes))
            test_classes.clear()
        if test_args.version == '1.0':
            schema_class = ObservedXMLSchema10 if test_args.inspect else XMLSchema10
            test_class = test_class_builder(test_file, test_args, test_num, schema_class, check_with_lxml)
        else:
            schema_class = ObservedXMLSchema11 if test_args.inspect else XMLSchema11
            test_class = test_class_builder(test_file, test_args, test_num, schema_class, check_with_lxml=False)
        test_classes[test_class.__name__] = test_class
        if verbosity == 2:
            print(f'Create case {test_class.__name__} for file {os.path.relpath(test_file)}')
        logger.debug('Add XSD %s test class %r.', test_args.version, test_class.__name__)
    if line_buffer:
        raise ValueError('Not completed line continuation at the end!')
    return test_classes