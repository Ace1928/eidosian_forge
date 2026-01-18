import datetime
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.cloudwatch import CloudWatchConnection
def test_build_put_params_multiple_everything(self):
    params = {}
    name = ['whatever', 'goeshere']
    value = None
    timestamp = [datetime.datetime(2013, 5, 13, 9, 2, 35), datetime.datetime(2013, 5, 12, 9, 2, 35)]
    unit = ['lbs', 'ft']
    dimensions = None
    statistics = [{'maximum': 5, 'minimum': 1, 'samplecount': 3, 'sum': 7}, {'maximum': 6, 'minimum': 2, 'samplecount': 4, 'sum': 5}]
    self.service_connection.build_put_params(params, name=name, value=value, timestamp=timestamp, unit=unit, dimensions=dimensions, statistics=statistics)
    self.assertEqual(params, {'MetricData.member.1.MetricName': 'whatever', 'MetricData.member.1.StatisticValues.Maximum': 5, 'MetricData.member.1.StatisticValues.Minimum': 1, 'MetricData.member.1.StatisticValues.SampleCount': 3, 'MetricData.member.1.StatisticValues.Sum': 7, 'MetricData.member.1.Timestamp': '2013-05-13T09:02:35', 'MetricData.member.1.Unit': 'lbs', 'MetricData.member.2.MetricName': 'goeshere', 'MetricData.member.2.StatisticValues.Maximum': 6, 'MetricData.member.2.StatisticValues.Minimum': 2, 'MetricData.member.2.StatisticValues.SampleCount': 4, 'MetricData.member.2.StatisticValues.Sum': 5, 'MetricData.member.2.Timestamp': '2013-05-12T09:02:35', 'MetricData.member.2.Unit': 'ft'})