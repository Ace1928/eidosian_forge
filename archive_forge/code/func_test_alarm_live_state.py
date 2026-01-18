import copy
import json
from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import aodh
from heat.engine.clients.os import octavia
from heat.engine import resource
from heat.engine.resources.openstack.aodh import alarm
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests import utils
def test_alarm_live_state(self):
    snippet = template_format.parse(alarm_template)
    self.stack = utils.parse_stack(snippet)
    resource_defns = self.stack.t.resource_definitions(self.stack)
    self.rsrc_defn = resource_defns['MEMAlarmHigh']
    self.client = mock.Mock()
    self.patchobject(alarm.AodhAlarm, 'client', return_value=self.client)
    alarm_res = alarm.AodhAlarm('alarm', self.rsrc_defn, self.stack)
    alarm_res.create()
    value = {'description': 'Scale-up if MEM > 50% for 1 minute', 'alarm_actions': [], 'time_constraints': [], 'threshold_rule': {'meter_name': 'MemoryUtilization', 'statistic': 'avg', 'period': '60', 'evaluation_periods': '1', 'threshold': '50', 'matching_metadata': {}, 'comparison_operator': 'gt', 'query': [{'field': 'c', 'op': 'ne', 'value': 'z'}]}}
    self.client.alarm.get.return_value = value
    expected_data = {'description': 'Scale-up if MEM > 50% for 1 minute', 'alarm_actions': [], 'statistic': 'avg', 'period': '60', 'evaluation_periods': '1', 'threshold': '50', 'matching_metadata': {}, 'comparison_operator': 'gt', 'query': [{'field': 'c', 'op': 'ne', 'value': 'z'}], 'repeat_actions': None, 'ok_actions': None, 'insufficient_data_actions': None, 'severity': None, 'enabled': None}
    reality = alarm_res.get_live_state(alarm_res.properties)
    self.assertEqual(expected_data, reality)