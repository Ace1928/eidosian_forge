from tests.compat import mock, unittest
from boto.ec2.address import Address
def test_endElement_sets_correct_attributes_with_values(self):
    for arguments in [('publicIp', '192.168.1.1', 'public_ip'), ('instanceId', 1, 'instance_id'), ('domain', 'some domain', 'domain'), ('allocationId', 1, 'allocation_id'), ('associationId', 1, 'association_id'), ('somethingRandom', 'somethingRandom', 'somethingRandom')]:
        self.check_that_attribute_has_been_set(arguments[0], arguments[1], arguments[2])