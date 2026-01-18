from unittest import mock
from urllib import parse as urlparse
from heat.common import template_format
from heat.engine.clients import client_plugin
from heat.engine import resource
from heat.engine.resources.openstack.zaqar import queue
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_create_default_name(self):
    t = template_format.parse(wp_template)
    del t['Resources']['MyQueue2']['Properties']['name']
    self.parse_stack(t)
    queue = self.stack['MyQueue2']
    name_match = utils.PhysName(self.stack.name, 'MyQueue2')
    self.fc.queue.side_effect = FakeQueue
    scheduler.TaskRunner(queue.create)()
    queue_name = queue.physical_resource_name()
    self.assertEqual(name_match, queue_name)
    self.fc.api_version = 2
    self.assertEqual('http://127.0.0.1:8888/v2/queues/' + queue_name, queue.FnGetAtt('href'))
    self.fc.queue.assert_called_once_with(name_match, auto_create=False)