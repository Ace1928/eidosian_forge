import datetime
import freezegun
import http.client
from oslo_config import fixture as config_fixture
from oslo_serialization import jsonutils
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import filtering
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import test_v3
def clean_up_service(self):
    """Clean up service test data from Identity Limit Test Cases."""
    for service in self.service_list:
        PROVIDERS.catalog_api.delete_service(service['id'])