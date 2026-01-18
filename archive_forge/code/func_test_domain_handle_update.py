from unittest import mock
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import resource
from heat.engine.resources.openstack.keystone import domain
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_domain_handle_update(self):
    self.test_domain.resource_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    prop_diff = {domain.KeystoneDomain.DESCRIPTION: 'Test Domain updated', domain.KeystoneDomain.ENABLED: False, domain.KeystoneDomain.NAME: 'test_domain_2'}
    self.test_domain.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    self.domains.update.assert_called_once_with(domain=self.test_domain.resource_id, description=prop_diff[domain.KeystoneDomain.DESCRIPTION], enabled=prop_diff[domain.KeystoneDomain.ENABLED], name='test_domain_2')