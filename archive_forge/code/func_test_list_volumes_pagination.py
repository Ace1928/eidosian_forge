from fixtures import TimeoutException
from testtools import content
from openstack import exceptions
from openstack.tests.functional import base
from openstack import utils
def test_list_volumes_pagination(self):
    """Test pagination for list volumes functionality"""
    volumes = []
    num_volumes = 8
    for i in range(num_volumes):
        name = self.getUniqueString()
        v = self.user_cloud.create_volume(display_name=name, size=1)
        volumes.append(v)
    self.addCleanup(self.cleanup, volumes)
    result = []
    for i in self.user_cloud.list_volumes():
        if i['name'] and i['name'].startswith(self.id()):
            result.append(i['id'])
    self.assertEqual(sorted([i['id'] for i in volumes]), sorted(result))