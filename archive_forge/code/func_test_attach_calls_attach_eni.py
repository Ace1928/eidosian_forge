from tests.compat import mock, unittest
from boto.exception import BotoClientError
from boto.ec2.networkinterface import NetworkInterfaceCollection
from boto.ec2.networkinterface import NetworkInterfaceSpecification
from boto.ec2.networkinterface import PrivateIPAddress
from boto.ec2.networkinterface import Attachment, NetworkInterface
def test_attach_calls_attach_eni(self):
    self.eni_one.connection = mock.Mock()
    self.eni_one.attach('instance_id', 11)
    self.eni_one.connection.attach_network_interface.assert_called_with('eni-1', 'instance_id', 11, dry_run=False)