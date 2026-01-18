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
def test_image_set_data_upload_and_not_activate_notification(self):
    insurance = {'called': False}

    def data_iterator():
        self.notifier.log = []
        yield 'abcde'
        yield 'fghij'
        self.image_proxy.extra_properties['os_glance_importing_to_stores'] = 'store2'
        insurance['called'] = True
    self.image_proxy.set_data(data_iterator(), 10)
    output_logs = self.notifier.get_logs()
    self.assertEqual(1, len(output_logs))
    output_log = output_logs[0]
    self.assertEqual('INFO', output_log['notification_type'])
    self.assertEqual('image.upload', output_log['event_type'])
    self.assertEqual(self.image.image_id, output_log['payload']['id'])
    self.assertTrue(insurance['called'])