import datetime
from fixtures import TimeoutException
from openstack import exceptions
from openstack.tests.functional import base
from openstack import utils
def test_list_all_servers_bad_permissions(self):
    self.assertRaises(exceptions.SDKException, self.user_cloud.list_servers, all_projects=True)