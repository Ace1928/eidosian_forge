import collections
import logging
from unittest import mock
import fixtures
from oslotest import base
from testtools import compat
from testtools import matchers
from testtools import testcase
from taskflow import exceptions
from taskflow.tests import fixtures as taskflow_fixtures
from taskflow.tests import utils
from taskflow.utils import misc
def assertFailuresRegexp(self, exc_class, pattern, callable_obj, *args, **kwargs):
    """Asserts the callable failed with the given exception and message."""
    try:
        with utils.wrap_all_failures():
            callable_obj(*args, **kwargs)
    except exceptions.WrappedFailure as e:
        self.assertThat(e, FailureRegexpMatcher(exc_class, pattern))