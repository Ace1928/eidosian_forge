import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
def test_noexc_func_context_manager(self):
    ignore_assertion_error = self._make_filter_func()
    with ignore_assertion_error:
        pass