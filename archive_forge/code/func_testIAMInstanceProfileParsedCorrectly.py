import base64
from tests.compat import unittest, mock
from tests.unit import AWSMockServiceTestCase
from boto.ec2.connection import EC2Connection
def testIAMInstanceProfileParsedCorrectly(self):
    ec2 = EC2Connection(aws_access_key_id='aws_access_key_id', aws_secret_access_key='aws_secret_access_key')
    mock_response = mock.Mock()
    mock_response.read.return_value = RUN_INSTANCE_RESPONSE
    mock_response.status = 200
    ec2.make_request = mock.Mock(return_value=mock_response)
    reservation = ec2.run_instances(image_id='ami-12345')
    self.assertEqual(len(reservation.instances), 1)
    instance = reservation.instances[0]
    self.assertEqual(instance.image_id, 'ami-ed65ba84')
    self.assertEqual(instance.id, 'i-ff0f1299')
    self.assertDictEqual(instance.instance_profile, {'arn': 'arn:aws:iam::ownerid:instance-profile/myinstanceprofile', 'id': 'iamid'})