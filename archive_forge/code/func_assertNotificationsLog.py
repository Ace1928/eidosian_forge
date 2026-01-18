import datetime
from unittest import mock
from oslo_serialization import jsonutils
import webob
import wsme
from glance.api import policy
from glance.api.v2 import metadef_namespaces as namespaces
from glance.api.v2 import metadef_objects as objects
from glance.api.v2 import metadef_properties as properties
from glance.api.v2 import metadef_resource_types as resource_types
from glance.api.v2 import metadef_tags as tags
import glance.gateway
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
def assertNotificationsLog(self, expected_events):
    output_logs = self.notifier.get_logs()
    expected_logs_count = len(expected_events)
    self.assertEqual(expected_logs_count, len(output_logs))
    for output_log, event in zip(output_logs, expected_events):
        self.assertEqual('INFO', output_log['notification_type'])
        self.assertEqual(event['type'], output_log['event_type'])
        self.assertLessEqual(event['payload'].items(), output_log['payload'].items())
    self.notifier.log = []