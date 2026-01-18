import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
def test_filter_method_context_manager(self):
    ignore_assertion_error = self._make_filter_method()
    with ignore_assertion_error:
        assert False, 'This is a test'