import uuid
from openstack import exceptions
from openstack.tests.functional import base
from openstack import utils
def test_find_zone_share_ignore_missing_false(self):
    self.assertRaises(exceptions.ResourceNotFound, self.operator_cloud.dns.find_zone_share, self.zone, 'bogus_id', ignore_missing=False)