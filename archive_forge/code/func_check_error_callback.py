import sys
import types
from testtools import TestCase
from testtools.matchers import (
from extras import (
def check_error_callback(test, function, arg, expected_error_count, expect_result):
    """General test template for error_callback argument.

    :param test: Test case instance.
    :param function: Either try_import or try_imports.
    :param arg: Name or names to import.
    :param expected_error_count: Expected number of calls to the callback.
    :param expect_result: Boolean for whether a module should
        ultimately be returned or not.
    """
    cb_calls = []

    def cb(e):
        test.assertIsInstance(e, ImportError)
        cb_calls.append(e)
    try:
        result = function(arg, error_callback=cb)
    except ImportError:
        test.assertFalse(expect_result)
    else:
        if expect_result:
            test.assertThat(result, Not(Is(None)))
        else:
            test.assertThat(result, Is(None))
    test.assertEquals(len(cb_calls), expected_error_count)