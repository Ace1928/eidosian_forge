from oslo_config import fixture as cfg_fixture
from oslo_messaging import conffixture as msg_fixture
from oslotest import createfile
import webob.dec
from keystonemiddleware import audit
from keystonemiddleware.tests.unit import utils
@staticmethod
def get_environ_header(req_type=None):
    env_headers = {'HTTP_X_SERVICE_CATALOG': '[{"endpoints_links": [],\n                            "endpoints": [{"adminURL":\n                                           "http://admin_host:8774",\n                                           "region": "RegionOne",\n                                           "publicURL":\n                                           "http://public_host:8774",\n                                           "internalURL":\n                                           "http://internal_host:8774",\n                                           "id": "resource_id"}],\n                            "type": "compute",\n                            "name": "nova"}]', 'HTTP_X_USER_ID': 'user_id', 'HTTP_X_USER_NAME': 'user_name', 'HTTP_X_AUTH_TOKEN': 'token', 'HTTP_X_PROJECT_ID': 'tenant_id', 'HTTP_X_IDENTITY_STATUS': 'Confirmed'}
    if req_type:
        env_headers['REQUEST_METHOD'] = req_type
    return env_headers