from tests.unit import unittest
from tests.compat import mock
from boto.ec2.elb import ELBConnection
from boto.ec2.elb import LoadBalancer
def test_next_token(self):
    elb = ELBConnection(aws_access_key_id='aws_access_key_id', aws_secret_access_key='aws_secret_access_key')
    mock_response = mock.Mock()
    mock_response.read.return_value = DISABLE_RESPONSE
    mock_response.status = 200
    elb.make_request = mock.Mock(return_value=mock_response)
    disabled = elb.disable_availability_zones('mine', ['sample-zone'])
    self.assertEqual(disabled, ['sample-zone'])