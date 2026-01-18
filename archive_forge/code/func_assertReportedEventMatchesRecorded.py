import datetime
from unittest import mock
import uuid
import freezegun
import http.client
from oslo_db import exception as oslo_db_exception
from oslo_utils import timeutils
from testtools import matchers
from keystone.common import provider_api
from keystone.common import utils
from keystone.models import revoke_model
from keystone.tests.unit import test_v3
def assertReportedEventMatchesRecorded(self, event, sample, before_time):
    after_time = timeutils.utcnow()
    event_issued_before = timeutils.normalize_time(timeutils.parse_isotime(event['issued_before']))
    self.assertLessEqual(before_time, event_issued_before, 'invalid event issued_before time; %s is not later than %s.' % (utils.isotime(event_issued_before, subsecond=True), utils.isotime(before_time, subsecond=True)))
    self.assertLessEqual(event_issued_before, after_time, 'invalid event issued_before time; %s is not earlier than %s.' % (utils.isotime(event_issued_before, subsecond=True), utils.isotime(after_time, subsecond=True)))
    del event['issued_before']
    del event['revoked_at']
    self.assertEqual(sample, event)