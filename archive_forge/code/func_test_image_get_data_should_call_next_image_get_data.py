import datetime
from unittest import mock
import glance_store
from oslo_config import cfg
import oslo_messaging
import webob
import glance.async_
from glance.common import exception
from glance.common import timeutils
import glance.context
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
from glance.tests import utils
def test_image_get_data_should_call_next_image_get_data(self):
    with mock.patch.object(self.image, 'get_data') as get_data_mock:
        self.image_proxy.get_data()
        self.assertTrue(get_data_mock.called)