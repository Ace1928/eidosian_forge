from boto.compat import json, map, six, zip
from boto.connection import AWSQueryConnection
from boto.ec2.cloudwatch.metric import Metric
from boto.ec2.cloudwatch.alarm import MetricAlarm, MetricAlarms, AlarmHistoryItem
from boto.ec2.cloudwatch.datapoint import Datapoint
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
def build_put_params(self, params, name, value=None, timestamp=None, unit=None, dimensions=None, statistics=None):
    args = (name, value, unit, dimensions, statistics, timestamp)
    length = max(map(lambda a: len(a) if isinstance(a, list) else 1, args))

    def aslist(a):
        if isinstance(a, list):
            if len(a) != length:
                raise Exception('Must specify equal number of elements; expected %d.' % length)
            return a
        return [a] * length
    for index, (n, v, u, d, s, t) in enumerate(zip(*map(aslist, args))):
        metric_data = {'MetricName': n}
        if timestamp:
            metric_data['Timestamp'] = t.isoformat()
        if unit:
            metric_data['Unit'] = u
        if dimensions:
            self.build_dimension_param(d, metric_data)
        if statistics:
            metric_data['StatisticValues.Maximum'] = s['maximum']
            metric_data['StatisticValues.Minimum'] = s['minimum']
            metric_data['StatisticValues.SampleCount'] = s['samplecount']
            metric_data['StatisticValues.Sum'] = s['sum']
            if value is not None:
                msg = 'You supplied a value and statistics for a ' + 'metric.Posting statistics and not value.'
                boto.log.warn(msg)
        elif value is not None:
            metric_data['Value'] = v
        else:
            raise Exception('Must specify a value or statistics to put.')
        for key, val in six.iteritems(metric_data):
            params['MetricData.member.%d.%s' % (index + 1, key)] = val