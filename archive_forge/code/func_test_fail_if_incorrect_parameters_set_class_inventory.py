import collections
import copy
import uuid
from osc_placement.tests.functional import base
def test_fail_if_incorrect_parameters_set_class_inventory(self):
    exc = self.assertRaises(base.CommandException, self.openstack, 'resource provider inventory class set')
    self.assertIn(base.ARGUMENTS_MISSING, str(exc))
    exc = self.assertRaises(base.CommandException, self.openstack, 'resource provider inventory class set fake_uuid')
    self.assertIn(base.ARGUMENTS_MISSING, str(exc))
    exc = self.assertRaises(base.CommandException, self.openstack, 'resource provider inventory class set fake_uuid fake_class --total 5 --unknown 1')
    self.assertIn('unrecognized arguments', str(exc))
    rp = self.resource_provider_create()
    exc = self.assertRaises(base.CommandException, self.openstack, 'resource provider inventory class set %s VCPU' % rp['uuid'])
    self.assertIn(base.ARGUMENTS_REQUIRED % '--total', str(exc))