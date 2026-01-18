import re
import argparse
import os
import fileinput
import logging
from xmlschema.cli import xsd_version_number, defuse_data
from xmlschema.validators import XMLSchema10, XMLSchema11
from ._observers import ObservedXMLSchema10, ObservedXMLSchema11
def get_test_line_args_parser():
    """Gets an arguments parser for uncommented on not blank "testfiles" lines."""
    parser = argparse.ArgumentParser(add_help=True)
    parser.usage = "TEST_FILE [OPTIONS]\nTry 'TEST_FILE --help' for more information."
    parser.add_argument('filename', metavar='TEST_FILE', type=str, help='Test filename (relative path).')
    parser.add_argument('-L', dest='locations', nargs=2, type=str, default=None, action='append', metavar='URI-URL', help='Schema location hint overrides.')
    parser.add_argument('--version', dest='version', metavar='VERSION', type=xsd_version_number, default='1.0', help='XSD schema version to use for the test case (default is 1.0).')
    parser.add_argument('--errors', type=int, default=0, metavar='NUM', help='Number of errors expected (default=0).')
    parser.add_argument('--warnings', type=int, default=0, metavar='NUM', help='Number of warnings expected (default=0).')
    parser.add_argument('--inspect', action='store_true', default=False, help='Inspect using an observed custom schema class.')
    parser.add_argument('--defuse', metavar='(always, remote, never)', type=defuse_data, default='remote', help='Define when to use the defused XML data loaders.')
    parser.add_argument('--timeout', type=int, default=300, metavar='SEC', help='Timeout for fetching resources (default=300).')
    parser.add_argument('--validation-only', action='store_true', default=False, help='Skip decode/encode tests on XML data.')
    parser.add_argument('--no-pickle', action='store_true', default=False, help='Skip pickling/unpickling test on schema (max recursion exceeded).')
    parser.add_argument('--lax-encode', action='store_true', default=False, help='Use lax mode on encode checks (for cases where test data uses default or fixed values or some test data are skipped by wildcards processContents). Ignored on schema tests.')
    parser.add_argument('--debug', action='store_true', default=False, help='Activate the debug mode (only the cases with --debug are executed).')
    parser.add_argument('--codegen', action='store_true', default=False, help='Test code generation with XML data bindings module. For default test code generation if the same command option is provided.')
    return parser