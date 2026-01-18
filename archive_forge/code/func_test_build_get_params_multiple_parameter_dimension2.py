import datetime
from boto.ec2.cloudwatch import CloudWatchConnection
from tests.compat import unittest, OrderedDict
def test_build_get_params_multiple_parameter_dimension2(self):
    self.maxDiff = None
    c = CloudWatchConnection()
    params = {}
    dimensions = OrderedDict((('D1', ['V1', 'V2']), ('D2', 'W'), ('D3', None)))
    c.build_dimension_param(dimensions, params)
    expected_params = {'Dimensions.member.1.Name': 'D1', 'Dimensions.member.1.Value': 'V1', 'Dimensions.member.2.Name': 'D1', 'Dimensions.member.2.Value': 'V2', 'Dimensions.member.3.Name': 'D2', 'Dimensions.member.3.Value': 'W', 'Dimensions.member.4.Name': 'D3'}
    self.assertEqual(params, expected_params)