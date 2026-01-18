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
def test_mem_alarm_high_correct_matching_metadata(self):
    t = template_format.parse(alarm_template)
    properties = t['Resources']['MEMAlarmHigh']['Properties']
    properties['matching_metadata'] = {'fro': 'bar', 'bro': True, 'dro': 1234, 'pro': '{"Mem": {"Ala": {"Hig"}}}', 'tro': [1, 2, 3, 4]}
    test_stack = self.create_stack(template=json.dumps(t))
    test_stack.create()
    rsrc = test_stack['MEMAlarmHigh']
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    rsrc.properties.data = rsrc.get_alarm_props(properties)
    self.assertIsNone(rsrc.properties.data.get('matching_metadata'))
    for key in rsrc.properties.data['threshold_rule']['query']:
        self.assertIsInstance(key['value'], str)