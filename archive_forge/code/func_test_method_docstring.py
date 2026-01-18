import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
def test_method_docstring(self):
    ignore_func = self._make_filter_method()
    self.assertEqual('Ignore some exceptions M.', ignore_func.__doc__)