import uuid
from openstack import exceptions
from openstack.tests.functional import base
def test_find_flavors_no_match_ignore_true(self):
    rslt = self.conn.compute.find_flavor('not a flavor', ignore_missing=True)
    self.assertIsNone(rslt)