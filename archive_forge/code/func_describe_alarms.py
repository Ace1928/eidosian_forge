from boto.compat import json, map, six, zip
from boto.connection import AWSQueryConnection
from boto.ec2.cloudwatch.metric import Metric
from boto.ec2.cloudwatch.alarm import MetricAlarm, MetricAlarms, AlarmHistoryItem
from boto.ec2.cloudwatch.datapoint import Datapoint
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
def describe_alarms(self, action_prefix=None, alarm_name_prefix=None, alarm_names=None, max_records=None, state_value=None, next_token=None):
    """
        Retrieves alarms with the specified names. If no name is specified, all
        alarms for the user are returned. Alarms can be retrieved by using only
        a prefix for the alarm name, the alarm state, or a prefix for any
        action.

        :type action_prefix: string
        :param action_prefix: The action name prefix.

        :type alarm_name_prefix: string
        :param alarm_name_prefix: The alarm name prefix. AlarmNames cannot
            be specified if this parameter is specified.

        :type alarm_names: list
        :param alarm_names: A list of alarm names to retrieve information for.

        :type max_records: int
        :param max_records: The maximum number of alarm descriptions
            to retrieve.

        :type state_value: string
        :param state_value: The state value to be used in matching alarms.

        :type next_token: string
        :param next_token: The token returned by a previous call to
            indicate that there is more data.

        :rtype list
        """
    params = {}
    if action_prefix:
        params['ActionPrefix'] = action_prefix
    if alarm_name_prefix:
        params['AlarmNamePrefix'] = alarm_name_prefix
    elif alarm_names:
        self.build_list_params(params, alarm_names, 'AlarmNames.member.%s')
    if max_records:
        params['MaxRecords'] = max_records
    if next_token:
        params['NextToken'] = next_token
    if state_value:
        params['StateValue'] = state_value
    result = self.get_list('DescribeAlarms', params, [('MetricAlarms', MetricAlarms)])
    ret = result[0]
    ret.next_token = result.next_token
    return ret