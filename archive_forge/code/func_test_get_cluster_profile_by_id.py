import time
from testtools import content
from openstack.tests.functional import base
def test_get_cluster_profile_by_id(self):
    profile_name = 'test_profile'
    spec = {'properties': {'flavor': self.flavor.name, 'image': self.image.name, 'networks': [{'network': 'private'}], 'security_groups': ['default']}, 'type': 'os.nova.server', 'version': 1.0}
    self.addDetail('profile', content.text_content(profile_name))
    profile = self.user_cloud.create_cluster_profile(name=profile_name, spec=spec)
    self.addCleanup(self.cleanup_profile, profile['id'])
    profile_get = self.user_cloud.get_cluster_profile_by_id(profile['id'])
    profile['created_at'] = 'ignore'
    profile_get['created_at'] = 'ignore'
    self.assertEqual(profile_get, profile)