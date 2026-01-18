import datetime
from boto.ec2.cloudwatch import CloudWatchConnection
from tests.compat import unittest, OrderedDict
def test_build_put_params_multiple_dimensions(self):
    c = CloudWatchConnection()
    params = {}
    c.build_put_params(params, name='N', value=[1, 2], dimensions=[{'D': 'V'}, {'D': 'W'}])
    expected_params = {'MetricData.member.1.MetricName': 'N', 'MetricData.member.1.Value': 1, 'MetricData.member.1.Dimensions.member.1.Name': 'D', 'MetricData.member.1.Dimensions.member.1.Value': 'V', 'MetricData.member.2.MetricName': 'N', 'MetricData.member.2.Value': 2, 'MetricData.member.2.Dimensions.member.1.Name': 'D', 'MetricData.member.2.Dimensions.member.1.Value': 'W'}
    self.assertEqual(params, expected_params)