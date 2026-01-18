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
def test_initiator_request_id(self):
    data = self.build_authentication_request(user_id=self.user_id, password=self.user['password'])
    self.post('/auth/tokens', body=data)
    audit = self._audits[-1]
    initiator = audit['payload']['initiator']
    self.assertIn('request_id', initiator)