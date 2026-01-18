from tests.compat import mock, unittest
from boto.exception import BotoClientError
from boto.ec2.networkinterface import NetworkInterfaceCollection
from boto.ec2.networkinterface import NetworkInterfaceSpecification
from boto.ec2.networkinterface import PrivateIPAddress
from boto.ec2.networkinterface import Attachment, NetworkInterface
def test_detach_calls_detach_network_interface(self):
    self.eni_one.connection = mock.Mock()
    self.eni_one.detach()
    self.eni_one.connection.detach_network_interface.assert_called_with('eni-attach-1', False, dry_run=False)