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
def test_delete_service(self):
    service_ref = unit.new_service_ref()
    PROVIDERS.catalog_api.create_service(service_ref['id'], service_ref)
    PROVIDERS.catalog_api.delete_service(service_ref['id'])
    self._assert_notify_sent(service_ref['id'], DELETED_OPERATION, 'service')
    self._assert_last_audit(service_ref['id'], DELETED_OPERATION, 'service', cadftaxonomy.SECURITY_SERVICE)