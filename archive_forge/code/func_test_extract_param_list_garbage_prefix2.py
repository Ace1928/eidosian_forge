from heat.api.aws import exception as aws_exception
from heat.api.aws import utils as api_utils
from heat.common import exception as common_exception
from heat.tests import common
def test_extract_param_list_garbage_prefix2(self):
    p = {'AMetricData.member.1.MetricName': 'foo', 'BMetricData.member.1.Unit': 'Bytes', 'CMetricData.member.1.Value': 234333}
    params = api_utils.extract_param_list(p, prefix='MetricData')
    self.assertEqual(0, len(params))