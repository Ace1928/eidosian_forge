from heat.api.aws import exception as aws_exception
from heat.api.aws import utils as api_utils
from heat.common import exception as common_exception
from heat.tests import common
def test_extract_param_list_garbage_suffix(self):
    p = {'MetricData.member.1.AMetricName': 'foo', 'MetricData.member.1.Unit': 'Bytes', 'MetricData.member.1.Value': 234333}
    params = api_utils.extract_param_list(p, prefix='MetricData')
    self.assertEqual(1, len(params))
    self.assertNotIn('MetricName', params[0])
    self.assertIn('Unit', params[0])
    self.assertIn('Value', params[0])
    self.assertEqual('Bytes', params[0]['Unit'])
    self.assertEqual(234333, params[0]['Value'])