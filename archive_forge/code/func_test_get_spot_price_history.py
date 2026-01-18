from tests.unit import AWSMockServiceTestCase
from boto.ec2.connection import EC2Connection
def test_get_spot_price_history(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.get_spot_price_history(instance_type='c3.large')
    self.assert_request_parameters({'Action': 'DescribeSpotPriceHistory', 'InstanceType': 'c3.large'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEqual(len(response), 2)
    self.assertEqual(response.next_token, 'q5GwEl5bMGjKq6YmhpDLJ7hEwyWU54jJC2GQ93n61vZV4s1+fzZ674xzvUlTihrl')
    self.assertEqual(response.nextToken, 'q5GwEl5bMGjKq6YmhpDLJ7hEwyWU54jJC2GQ93n61vZV4s1+fzZ674xzvUlTihrl')
    self.assertEqual(response[0].instance_type, 'c3.large')
    self.assertEqual(response[0].availability_zone, 'us-west-2c')
    self.assertEqual(response[1].instance_type, 'c3.large')
    self.assertEqual(response[1].availability_zone, 'us-west-2b')
    response = self.service_connection.get_spot_price_history(filters={'instance-type': 'c3.large'})
    self.assert_request_parameters({'Action': 'DescribeSpotPriceHistory', 'Filter.1.Name': 'instance-type', 'Filter.1.Value.1': 'c3.large'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    response = self.service_connection.get_spot_price_history(next_token='foobar')
    self.assert_request_parameters({'Action': 'DescribeSpotPriceHistory', 'NextToken': 'foobar'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])