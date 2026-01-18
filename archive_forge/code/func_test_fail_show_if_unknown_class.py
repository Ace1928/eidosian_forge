import uuid
from osc_placement.tests.functional import base
def test_fail_show_if_unknown_class(self):
    self.assertCommandFailed('No such resource class', self.resource_class_show, 'UNKNOWN')