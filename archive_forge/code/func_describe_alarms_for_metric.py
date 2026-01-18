from boto.compat import json, map, six, zip
from boto.connection import AWSQueryConnection
from boto.ec2.cloudwatch.metric import Metric
from boto.ec2.cloudwatch.alarm import MetricAlarm, MetricAlarms, AlarmHistoryItem
from boto.ec2.cloudwatch.datapoint import Datapoint
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
def describe_alarms_for_metric(self, metric_name, namespace, period=None, statistic=None, dimensions=None, unit=None):
    """
        Retrieves all alarms for a single metric. Specify a statistic, period,
        or unit to filter the set of alarms further.

        :type metric_name: string
        :param metric_name: The name of the metric.

        :type namespace: string
        :param namespace: The namespace of the metric.

        :type period: int
        :param period: The period in seconds over which the statistic
            is applied.

        :type statistic: string
        :param statistic: The statistic for the metric.

        :type dimensions: dict
        :param dimensions: A dictionary containing name/value
            pairs that will be used to filter the results. The key in
            the dictionary is the name of a Dimension. The value in
            the dictionary is either a scalar value of that Dimension
            name that you want to filter on, a list of values to
            filter on or None if you want all metrics with that
            Dimension name.

        :type unit: string

        :rtype list
        """
    params = {'MetricName': metric_name, 'Namespace': namespace}
    if period:
        params['Period'] = period
    if statistic:
        params['Statistic'] = statistic
    if dimensions:
        self.build_dimension_param(dimensions, params)
    if unit:
        params['Unit'] = unit
    return self.get_list('DescribeAlarmsForMetric', params, [('member', MetricAlarm)])