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
def test_create_endpoint(self):
    endpoint_ref = unit.new_endpoint_ref(service_id=self.service_id, interface='public', region_id=self.region_id)
    PROVIDERS.catalog_api.create_endpoint(endpoint_ref['id'], endpoint_ref)
    self._assert_notify_sent(endpoint_ref['id'], CREATED_OPERATION, 'endpoint')
    self._assert_last_audit(endpoint_ref['id'], CREATED_OPERATION, 'endpoint', cadftaxonomy.SECURITY_ENDPOINT)