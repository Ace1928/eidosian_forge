import sys
import warnings
from oslo_log import log
from sqlalchemy import exc
from testtools import matchers
from keystone.tests import unit
def test_deprecation_warnings_are_raised_as_exceptions_in_tests(self):
    self.assertThat(lambda: warnings.warn('this is deprecated', DeprecationWarning), matchers.raises(DeprecationWarning))