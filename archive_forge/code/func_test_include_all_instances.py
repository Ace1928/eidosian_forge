from tests.compat import mock, unittest
from boto.ec2.connection import EC2Connection
def test_include_all_instances(self):
    ec2 = EC2Connection(aws_access_key_id='aws_access_key_id', aws_secret_access_key='aws_secret_access_key')
    mock_response = mock.Mock()
    mock_response.read.return_value = INSTANCE_STATUS_RESPONSE
    mock_response.status = 200
    ec2.make_request = mock.Mock(return_value=mock_response)
    all_statuses = ec2.get_all_instance_status(include_all_instances=True)
    self.assertIn('IncludeAllInstances', ec2.make_request.call_args[0][1])
    self.assertEqual('true', ec2.make_request.call_args[0][1]['IncludeAllInstances'])
    self.assertEqual(all_statuses.next_token, 'page-2')