from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import aodh
from heat.engine.resources.openstack.aodh.gnocchi import alarm as gnocchi
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_gnocchi_alarm_aggr_by_metrics_live_state(self):
    snippet = template_format.parse(gnocchi_aggregation_by_metrics_alarm_template)
    self.stack = utils.parse_stack(snippet)
    resource_defns = self.stack.t.resource_definitions(self.stack)
    self.rsrc_defn = resource_defns['GnoAggregationByMetricsAlarm']
    self.client = mock.Mock()
    self.patchobject(gnocchi.AodhGnocchiAggregationByMetricsAlarm, 'client', return_value=self.client)
    alarm_res = gnocchi.AodhGnocchiAggregationByMetricsAlarm('alarm', self.rsrc_defn, self.stack)
    alarm_res.create()
    value = {'description': 'Do stuff with gnocchi metrics', 'alarm_actions': [], 'time_constraints': [], 'gnocchi_aggregation_by_metrics_threshold_rule': {'metrics': ['911fce07-e0d7-4210-8c8c-4a9d811fcabc', '2543d435-fe93-4443-9351-fb0156930f94'], 'evaluation_periods': 1, 'aggregation_method': 'mean', 'granularity': 60, 'threshold': 50, 'comparison_operator': 'gt'}}
    self.client.alarm.get.return_value = value
    expected_data = {'description': 'Do stuff with gnocchi metrics', 'alarm_actions': [], 'metrics': ['911fce07-e0d7-4210-8c8c-4a9d811fcabc', '2543d435-fe93-4443-9351-fb0156930f94'], 'evaluation_periods': 1, 'aggregation_method': 'mean', 'granularity': 60, 'threshold': 50, 'comparison_operator': 'gt', 'insufficient_data_actions': None, 'enabled': None, 'ok_actions': None, 'repeat_actions': None, 'severity': None}
    reality = alarm_res.get_live_state(alarm_res.properties)
    self.assertEqual(expected_data, reality)