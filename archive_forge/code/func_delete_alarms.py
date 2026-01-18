from boto.compat import json, map, six, zip
from boto.connection import AWSQueryConnection
from boto.ec2.cloudwatch.metric import Metric
from boto.ec2.cloudwatch.alarm import MetricAlarm, MetricAlarms, AlarmHistoryItem
from boto.ec2.cloudwatch.datapoint import Datapoint
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
def delete_alarms(self, alarms):
    """
        Deletes all specified alarms. In the event of an error, no
        alarms are deleted.

        :type alarms: list
        :param alarms: List of alarm names.
        """
    params = {}
    self.build_list_params(params, alarms, 'AlarmNames.member.%s')
    return self.get_status('DeleteAlarms', params)