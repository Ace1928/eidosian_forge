from boto.compat import json, map, six, zip
from boto.connection import AWSQueryConnection
from boto.ec2.cloudwatch.metric import Metric
from boto.ec2.cloudwatch.alarm import MetricAlarm, MetricAlarms, AlarmHistoryItem
from boto.ec2.cloudwatch.datapoint import Datapoint
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
def put_metric_alarm(self, alarm):
    """
        Creates or updates an alarm and associates it with the specified Amazon
        CloudWatch metric. Optionally, this operation can associate one or more
        Amazon Simple Notification Service resources with the alarm.

        When this operation creates an alarm, the alarm state is immediately
        set to INSUFFICIENT_DATA. The alarm is evaluated and its StateValue is
        set appropriately. Any actions associated with the StateValue is then
        executed.

        When updating an existing alarm, its StateValue is left unchanged.

        :type alarm: boto.ec2.cloudwatch.alarm.MetricAlarm
        :param alarm: MetricAlarm object.
        """
    params = {'AlarmName': alarm.name, 'MetricName': alarm.metric, 'Namespace': alarm.namespace, 'Statistic': alarm.statistic, 'ComparisonOperator': alarm.comparison, 'Threshold': alarm.threshold, 'EvaluationPeriods': alarm.evaluation_periods, 'Period': alarm.period}
    if alarm.actions_enabled is not None:
        params['ActionsEnabled'] = alarm.actions_enabled
    if alarm.alarm_actions:
        self.build_list_params(params, alarm.alarm_actions, 'AlarmActions.member.%s')
    if alarm.description:
        params['AlarmDescription'] = alarm.description
    if alarm.dimensions:
        self.build_dimension_param(alarm.dimensions, params)
    if alarm.insufficient_data_actions:
        self.build_list_params(params, alarm.insufficient_data_actions, 'InsufficientDataActions.member.%s')
    if alarm.ok_actions:
        self.build_list_params(params, alarm.ok_actions, 'OKActions.member.%s')
    if alarm.unit:
        params['Unit'] = alarm.unit
    alarm.connection = self
    return self.get_status('PutMetricAlarm', params)