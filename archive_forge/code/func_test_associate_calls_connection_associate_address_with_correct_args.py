from tests.compat import mock, unittest
from boto.ec2.address import Address
def test_associate_calls_connection_associate_address_with_correct_args(self):
    self.address.associate(network_interface_id=1)
    self.address.connection.associate_address.assert_called_with(instance_id=None, public_ip='192.168.1.1', network_interface_id=1, private_ip_address=None, allocation_id='aid1', allow_reassociation=False, dry_run=False)