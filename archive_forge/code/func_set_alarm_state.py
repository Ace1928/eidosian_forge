from boto.compat import json, map, six, zip
from boto.connection import AWSQueryConnection
from boto.ec2.cloudwatch.metric import Metric
from boto.ec2.cloudwatch.alarm import MetricAlarm, MetricAlarms, AlarmHistoryItem
from boto.ec2.cloudwatch.datapoint import Datapoint
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
def set_alarm_state(self, alarm_name, state_reason, state_value, state_reason_data=None):
    """
        Temporarily sets the state of an alarm. When the updated StateValue
        differs from the previous value, the action configured for the
        appropriate state is invoked. This is not a permanent change. The next
        periodic alarm check (in about a minute) will set the alarm to its
        actual state.

        :type alarm_name: string
        :param alarm_name: Descriptive name for alarm.

        :type state_reason: string
        :param state_reason: Human readable reason.

        :type state_value: string
        :param state_value: OK | ALARM | INSUFFICIENT_DATA

        :type state_reason_data: string
        :param state_reason_data: Reason string (will be jsonified).
        """
    params = {'AlarmName': alarm_name, 'StateReason': state_reason, 'StateValue': state_value}
    if state_reason_data:
        params['StateReasonData'] = json.dumps(state_reason_data)
    return self.get_status('SetAlarmState', params)