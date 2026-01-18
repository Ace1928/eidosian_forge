import io
import logging
from unittest import mock
from oslotest import base as test_base
from oslo_log import rate_limit
def test_install_twice(self):
    rate_limit.install_filter(100, 1)
    self.assertRaises(RuntimeError, rate_limit.install_filter, 100, 1)