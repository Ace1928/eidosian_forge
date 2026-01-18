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
def test_image_member_add_notification(self):
    self.image_member_repo_proxy.add(self.image_member_proxy)
    output_logs = self.notifier.get_logs()
    self.assertEqual(1, len(output_logs))
    output_log = output_logs[0]
    self.assertEqual('INFO', output_log['notification_type'])
    self.assertEqual('image.member.create', output_log['event_type'])
    self._assert_image_member_with_notifier(output_log)