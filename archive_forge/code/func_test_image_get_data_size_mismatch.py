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
def test_image_get_data_size_mismatch(self):
    self.image_proxy.size = 11
    list(self.image_proxy.get_data())
    output_logs = self.notifier.get_logs()
    self.assertEqual(1, len(output_logs))
    output_log = output_logs[0]
    self.assertEqual('ERROR', output_log['notification_type'])
    self.assertEqual('image.send', output_log['event_type'])
    self.assertEqual(self.image.image_id, output_log['payload']['image_id'])