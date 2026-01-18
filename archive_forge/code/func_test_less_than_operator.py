import io
import tempfile
from unittest import mock
import glance_store as store
from glance_store._drivers import cinder
from oslo_config import cfg
from oslo_log import log as logging
import webob
from glance.common import exception
from glance.common import store_utils
from glance.common import utils
from glance.tests.unit import base
from glance.tests import utils as test_utils
def test_less_than_operator(self):
    self.assertTrue(utils.evaluate_filter_op(9, 'lt', 10))
    self.assertFalse(utils.evaluate_filter_op(10, 'lt', 10))
    self.assertFalse(utils.evaluate_filter_op(11, 'lt', 10))