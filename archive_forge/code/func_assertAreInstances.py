import copy
from openstack.clustering.v1 import cluster
from openstack.tests.unit import base
def assertAreInstances(self, elements, elem_type):
    for e in elements:
        self.assertIsInstance(e, elem_type)