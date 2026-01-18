from boto.compat import json, map, six, zip
from boto.connection import AWSQueryConnection
from boto.ec2.cloudwatch.metric import Metric
from boto.ec2.cloudwatch.alarm import MetricAlarm, MetricAlarms, AlarmHistoryItem
from boto.ec2.cloudwatch.datapoint import Datapoint
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
def put_metric_data(self, namespace, name, value=None, timestamp=None, unit=None, dimensions=None, statistics=None):
    """
        Publishes metric data points to Amazon CloudWatch. Amazon Cloudwatch
        associates the data points with the specified metric. If the specified
        metric does not exist, Amazon CloudWatch creates the metric. If a list
        is specified for some, but not all, of the arguments, the remaining
        arguments are repeated a corresponding number of times.

        :type namespace: str
        :param namespace: The namespace of the metric.

        :type name: str or list
        :param name: The name of the metric.

        :type value: float or list
        :param value: The value for the metric.

        :type timestamp: datetime or list
        :param timestamp: The time stamp used for the metric. If not specified,
            the default value is set to the time the metric data was received.

        :type unit: string or list
        :param unit: The unit of the metric.  Valid Values: Seconds |
            Microseconds | Milliseconds | Bytes | Kilobytes |
            Megabytes | Gigabytes | Terabytes | Bits | Kilobits |
            Megabits | Gigabits | Terabits | Percent | Count |
            Bytes/Second | Kilobytes/Second | Megabytes/Second |
            Gigabytes/Second | Terabytes/Second | Bits/Second |
            Kilobits/Second | Megabits/Second | Gigabits/Second |
            Terabits/Second | Count/Second | None

        :type dimensions: dict
        :param dimensions: Add extra name value pairs to associate
            with the metric, i.e.:
            {'name1': value1, 'name2': (value2, value3)}

        :type statistics: dict or list
        :param statistics: Use a statistic set instead of a value, for example::

            {'maximum': 30, 'minimum': 1, 'samplecount': 100, 'sum': 10000}
        """
    params = {'Namespace': namespace}
    self.build_put_params(params, name, value=value, timestamp=timestamp, unit=unit, dimensions=dimensions, statistics=statistics)
    return self.get_status('PutMetricData', params, verb='POST')