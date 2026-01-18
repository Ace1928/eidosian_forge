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
def test_group_domain_grant(self):
    group_ref = unit.new_group_ref(domain_id=self.domain_id)
    group = PROVIDERS.identity_api.create_group(group_ref)
    PROVIDERS.identity_api.add_user_to_group(self.user_id, group['id'])
    url = '/domains/%s/groups/%s/roles/%s' % (self.domain_id, group['id'], self.role_id)
    self._test_role_assignment(url, self.role_id, domain=self.domain_id, group=group['id'])