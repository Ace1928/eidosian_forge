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
def test_image_set_data_duplicate(self):

    def data_iterator():
        self.notifier.log = []
        yield 'abcde'
        raise exception.Duplicate('Cant have duplicates')
    self.assertRaises(webob.exc.HTTPConflict, self.image_proxy.set_data, data_iterator(), 10)
    output_logs = self.notifier.get_logs()
    self.assertEqual(1, len(output_logs))
    output_log = output_logs[0]
    self.assertEqual('ERROR', output_log['notification_type'])
    self.assertEqual('image.upload', output_log['event_type'])
    self.assertIn('Cant have duplicates', output_log['payload'])