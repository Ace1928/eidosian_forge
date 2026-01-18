from unittest import mock
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.keystone import group
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_group_handle_update_default(self):
    self.test_group.resource_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    prop_diff = {group.KeystoneGroup.DESCRIPTION: 'Test Project updated'}
    self.test_group.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    self.groups.update.assert_called_once_with(group=self.test_group.resource_id, name=None, description=prop_diff[group.KeystoneGroup.DESCRIPTION], domain_id='default')