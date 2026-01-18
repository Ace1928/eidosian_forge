import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_config import fixture as config_fixture
from oslo_log import log
import oslo_messaging
from pycadf import cadftaxonomy
from pycadf import cadftype
from pycadf import eventfactory
from pycadf import resource as cadfresource
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_initiator_id_always_matches_user_id(self):
    while self._notifications:
        self._notifications.pop()
    self.get_scoped_token()
    self.assertEqual(len(self._notifications), 1)
    note = self._notifications.pop()
    initiator = note['initiator']
    self.assertEqual(self.user_id, initiator.id)
    self.assertEqual(self.user_id, initiator.user_id)