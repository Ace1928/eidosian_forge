from heat.api.aws import exception as aws_exception
from heat.api.aws import utils as api_utils
from heat.common import exception as common_exception
from heat.tests import common
def test_extract_param_list_multiple_missing(self):
    p = {'MetricData.member.1.MetricName': 'foo', 'MetricData.member.1.Unit': 'Bytes', 'MetricData.member.1.Value': 234333, 'MetricData.member.3.MetricName': 'foo2', 'MetricData.member.3.Unit': 'Bytes', 'MetricData.member.3.Value': 12345}
    params = api_utils.extract_param_list(p, prefix='MetricData')
    self.assertEqual(2, len(params))
    self.assertIn('MetricName', params[0])
    self.assertIn('MetricName', params[1])
    self.assertEqual('foo', params[0]['MetricName'])
    self.assertEqual('Bytes', params[0]['Unit'])
    self.assertEqual(234333, params[0]['Value'])
    self.assertEqual('foo2', params[1]['MetricName'])
    self.assertEqual('Bytes', params[1]['Unit'])
    self.assertEqual(12345, params[1]['Value'])