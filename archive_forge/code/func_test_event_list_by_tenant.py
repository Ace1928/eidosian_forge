from unittest import mock
from oslo_config import cfg
from oslo_messaging import conffixture
from heat.common import context
from heat.engine import resource as res
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine import service
from heat.engine import stack as parser
from heat.objects import event as event_object
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
@tools.stack_context('service_event_list_by_tenant')
def test_event_list_by_tenant(self):
    events = self.eng.list_events(self.ctx, None)
    self.assertEqual(4, len(events))
    for ev in events:
        self.assertIn('event_identity', ev)
        self.assertIsInstance(ev['event_identity'], dict)
        self.assertTrue(ev['event_identity']['path'].rsplit('/', 1)[1])
        self.assertIn('resource_name', ev)
        self.assertIn(ev['resource_name'], ('WebServer', 'service_event_list_by_tenant'))
        self.assertIn('physical_resource_id', ev)
        self.assertEqual('CREATE', ev['resource_action'])
        self.assertIn(ev['resource_status'], ('IN_PROGRESS', 'COMPLETE'))
        self.assertIn('resource_status_reason', ev)
        self.assertIn(ev['resource_status_reason'], ('state changed', 'Stack CREATE started', 'Stack CREATE completed successfully'))
        self.assertIn('resource_type', ev)
        self.assertIn(ev['resource_type'], ('AWS::EC2::Instance', 'OS::Heat::Stack'))
        self.assertIn('stack_identity', ev)
        self.assertIn('stack_name', ev)
        self.assertEqual(self.stack.name, ev['stack_name'])
        self.assertIn('event_time', ev)