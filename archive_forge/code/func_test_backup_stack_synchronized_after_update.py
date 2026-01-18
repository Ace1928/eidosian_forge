import copy
from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.db import api as db_api
from heat.engine.clients.os.keystone import fake_keystoneclient
from heat.engine import environment
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import service
from heat.engine import stack
from heat.engine import support
from heat.engine import template
from heat.objects import stack as stack_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_backup_stack_synchronized_after_update(self):
    """Test when backup stack updated correctly during stack update.

        Test checks the following scenario:
        1. Create stack
        2. Update stack (failed - so the backup should not be deleted)
        3. Update stack (failed - so the backup from step 2 should be updated)
        The test checks that backup stack is synchronized with the main stack.
        """
    tmpl_create = {'heat_template_version': '2013-05-23', 'resources': {'Ares': {'type': 'GenericResourceType'}}}
    self.stack = stack.Stack(self.ctx, 'test_update_stack_backup', template.Template(tmpl_create), disable_rollback=True)
    self.stack.store()
    self.stack.create()
    self.assertEqual((stack.Stack.CREATE, stack.Stack.COMPLETE), self.stack.state)
    tmpl_update = {'heat_template_version': '2013-05-23', 'resources': {'Ares': {'type': 'GenericResourceType'}, 'Bres': {'type': 'ResWithComplexPropsAndAttrs', 'properties': {'an_int': 0}}, 'Cres': {'type': 'ResourceWithPropsType', 'properties': {'Foo': {'get_resource': 'Bres'}}}}}
    self.patchobject(generic_rsrc.ResourceWithProps, 'handle_create', side_effect=[Exception, Exception])
    stack_with_new_resource = stack.Stack(self.ctx, 'test_update_stack_backup', template.Template(tmpl_update))
    self.stack.update(stack_with_new_resource)
    self.assertEqual((stack.Stack.UPDATE, stack.Stack.FAILED), self.stack.state)
    self.assertIn('Bres', self.stack._backup_stack())
    self.stack['Bres'].data_set('test', '42')
    tmpl_update['resources']['Bres']['properties']['an_int'] = 1
    updated_stack_second = stack.Stack(self.ctx, 'test_update_stack_backup', template.Template(tmpl_update))
    self.stack.update(updated_stack_second)
    self.assertEqual((stack.Stack.UPDATE, stack.Stack.FAILED), self.stack.state)
    backup = self.stack._backup_stack()
    self.assertEqual(1, backup['Bres'].properties['an_int'])
    self.assertEqual({}, backup['Bres'].data())
    self.assertEqual({'test': '42'}, self.stack['Bres'].data())