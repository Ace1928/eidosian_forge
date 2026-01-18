import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
import pprint
import traceback
import types
from datetime import datetime
from operator import itemgetter
import itertools
from functools import wraps
from contextlib import contextmanager
def assertRunTestResults(self, run_tests_report, expected_parse_results=None, msg=None):
    """
            Unit test assertion to evaluate output of ParserElement.runTests(). If a list of
            list-dict tuples is given as the expected_parse_results argument, then these are zipped
            with the report tuples returned by runTests and evaluated using assertParseResultsEquals.
            Finally, asserts that the overall runTests() success value is True.

            :param run_tests_report: tuple(bool, [tuple(str, ParseResults or Exception)]) returned from runTests
            :param expected_parse_results (optional): [tuple(str, list, dict, Exception)]
            """
    run_test_success, run_test_results = run_tests_report
    if expected_parse_results is not None:
        merged = [(rpt[0], rpt[1], expected) for rpt, expected in zip(run_test_results, expected_parse_results)]
        for test_string, result, expected in merged:
            fail_msg = next((exp for exp in expected if isinstance(exp, str)), None)
            expected_exception = next((exp for exp in expected if isinstance(exp, type) and issubclass(exp, Exception)), None)
            if expected_exception is not None:
                with self.assertRaises(expected_exception=expected_exception, msg=fail_msg or msg):
                    if isinstance(result, Exception):
                        raise result
            else:
                expected_list = next((exp for exp in expected if isinstance(exp, list)), None)
                expected_dict = next((exp for exp in expected if isinstance(exp, dict)), None)
                if (expected_list, expected_dict) != (None, None):
                    self.assertParseResultsEquals(result, expected_list=expected_list, expected_dict=expected_dict, msg=fail_msg or msg)
                else:
                    print('no validation for {!r}'.format(test_string))
    self.assertTrue(run_test_success, msg=msg if msg is not None else 'failed runTests')