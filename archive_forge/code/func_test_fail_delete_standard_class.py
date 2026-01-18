import uuid
from osc_placement.tests.functional import base
def test_fail_delete_standard_class(self):
    self.assertCommandFailed('Cannot delete standard resource class', self.resource_class_delete, 'VCPU')