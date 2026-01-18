from __future__ import absolute_import, print_function, division
import pytest
from petl.test.helpers import eq_, ieq, get_env_vars_named
def test_helper_get_env_vars_named_not_found(setup_helpers_get_env_vars_named):
    expected = None
    found = get_env_vars_named('PETL_TEST_HELPER_ENVVAR_NOT_FOUND_')
    eq_(found, expected)