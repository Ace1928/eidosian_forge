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
def test_mem_alarm_high_update_replace(self):
    """Tests resource replacing when changing non-updatable properties."""
    t = template_format.parse(alarm_template)
    properties = t['Resources']['MEMAlarmHigh']['Properties']
    properties['alarm_actions'] = ['signal_handler']
    properties['matching_metadata'] = {'a': 'v'}
    test_stack = self.create_stack(template=json.dumps(t))
    test_stack.create()
    rsrc = test_stack['MEMAlarmHigh']
    properties = copy.copy(rsrc.properties.data)
    properties['meter_name'] = 'temp'
    snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), properties)
    updater = scheduler.TaskRunner(rsrc.update, snippet)
    self.assertRaises(resource.UpdateReplace, updater)