import uuid
from keystoneauth1 import exceptions as ks_exc
import requests.exceptions
from openstack.config import cloud_region
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_no_adapter_opts_ignore_service_type(self):
    """Ignore unregistered conf section if service type not requested."""
    self.oslo_config_dict['heat'] = None
    self.assert_service_disabled('orchestration', 'Not in the list of requested service_types.', service_types=['compute'])