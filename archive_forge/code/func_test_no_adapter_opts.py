import uuid
from keystoneauth1 import exceptions as ks_exc
import requests.exceptions
from openstack.config import cloud_region
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_no_adapter_opts(self):
    """Conf section present, but opts for service type not registered."""
    self.oslo_config_dict['heat'] = None
    self.assert_service_disabled('orchestration', "Encountered an exception attempting to process config for project 'heat' (service type 'orchestration'): no such option")