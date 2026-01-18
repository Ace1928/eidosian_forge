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
def test_add_role_to_user_and_project(self):
    project_ref = unit.new_project_ref(self.domain_id)
    project = PROVIDERS.resource_api.create_project(project_ref['id'], project_ref)
    project_id = project['id']
    PROVIDERS.assignment_api.add_role_to_user_and_project(self.user_id, project_id, self.role_id)
    self.assertTrue(self._notifications)
    note = self._notifications[-1]
    self.assertEqual('created.role_assignment', note['action'])
    self.assertTrue(note['send_notification_called'])
    self._assert_event(self.role_id, project=project_id, user=self.user_id)