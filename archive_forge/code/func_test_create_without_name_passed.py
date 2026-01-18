from unittest import mock
from heat.common import template_format
from heat.engine.clients.os import sahara
from heat.engine.resources.openstack.sahara import job
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_create_without_name_passed(self):
    props = self.stack.t.t['resources']['job']['properties']
    del props['name']
    self.rsrc_defn = self.rsrc_defn.freeze(properties=props)
    jb = self._create_resource('job', self.rsrc_defn, self.stack, True)
    args = self.client.jobs.create.call_args[1]
    expected_args = {'name': 'fake_phys_name', 'type': 'MapReduce', 'libs': ['some res id'], 'description': 'test_description', 'is_public': True, 'is_protected': False, 'mains': []}
    self.assertEqual(expected_args, args)
    self.assertEqual('fake-resource-id', jb.resource_id)
    expected_state = (jb.CREATE, jb.COMPLETE)
    self.assertEqual(expected_state, jb.state)