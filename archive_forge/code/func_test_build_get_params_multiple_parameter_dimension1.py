import datetime
from boto.ec2.cloudwatch import CloudWatchConnection
from tests.compat import unittest, OrderedDict
def test_build_get_params_multiple_parameter_dimension1(self):
    self.maxDiff = None
    c = CloudWatchConnection()
    params = {}
    dimensions = OrderedDict((('D1', 'V'), ('D2', 'W')))
    c.build_dimension_param(dimensions, params)
    expected_params = {'Dimensions.member.1.Name': 'D1', 'Dimensions.member.1.Value': 'V', 'Dimensions.member.2.Name': 'D2', 'Dimensions.member.2.Value': 'W'}
    self.assertEqual(params, expected_params)