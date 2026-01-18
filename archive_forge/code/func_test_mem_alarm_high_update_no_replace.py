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
def test_mem_alarm_high_update_no_replace(self):
    """Tests update updatable properties without replacing the Alarm."""
    t = template_format.parse(alarm_template)
    properties = t['Resources']['MEMAlarmHigh']['Properties']
    properties['alarm_actions'] = ['signal_handler']
    properties['matching_metadata'] = {'a': 'v'}
    properties['query'] = [dict(field='b', op='eq', value='w')]
    test_stack = self.create_stack(template=json.dumps(t))
    update_mock = self.patchobject(self.fa.alarm, 'update')
    test_stack.create()
    rsrc = test_stack['MEMAlarmHigh']
    update_props = copy.deepcopy(rsrc.properties.data)
    update_props.update({'comparison_operator': 'lt', 'description': 'fruity', 'evaluation_periods': '2', 'period': '90', 'enabled': True, 'repeat_actions': True, 'statistic': 'max', 'threshold': '39', 'insufficient_data_actions': [], 'alarm_actions': [], 'ok_actions': ['signal_handler'], 'matching_metadata': {'x': 'y'}, 'query': [dict(field='c', op='ne', value='z')]})
    snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), update_props)
    scheduler.TaskRunner(rsrc.update, snippet)()
    self.assertEqual((rsrc.UPDATE, rsrc.COMPLETE), rsrc.state)
    self.assertEqual(1, update_mock.call_count)