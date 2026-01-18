from keystoneauth1 import session
from osc_lib import exceptions as osc_lib_exceptions
from requests_mock.contrib import fixture
from openstackclient.api import compute_v2 as compute
from openstackclient.tests.unit import utils
def test_host_show(self):
    FAKE_RESOURCE_1 = {'cpu': 2, 'disk_gb': 1028, 'host': 'c1a7de0ac9d94e4baceae031d05caae3', 'memory_mb': 8192, 'project': '(total)'}
    FAKE_RESOURCE_2 = {'cpu': 0, 'disk_gb': 0, 'host': 'c1a7de0ac9d94e4baceae031d05caae3', 'memory_mb': 512, 'project': '(used_now)'}
    FAKE_RESOURCE_3 = {'cpu': 0, 'disk_gb': 0, 'host': 'c1a7de0ac9d94e4baceae031d05caae3', 'memory_mb': 0, 'project': '(used_max)'}
    FAKE_HOST_RESP = [{'resource': FAKE_RESOURCE_1}, {'resource': FAKE_RESOURCE_2}, {'resource': FAKE_RESOURCE_3}]
    FAKE_HOST_LIST = [FAKE_RESOURCE_1, FAKE_RESOURCE_2, FAKE_RESOURCE_3]
    self.requests_mock.register_uri('GET', FAKE_URL + '/os-hosts/myhost', json={'host': FAKE_HOST_RESP}, status_code=200)
    ret = self.api.host_show(host='myhost')
    self.assertEqual(FAKE_HOST_LIST, ret)