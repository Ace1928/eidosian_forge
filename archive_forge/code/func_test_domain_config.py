import uuid
from openstack.identity.v3 import domain as _domain
from openstack.identity.v3 import domain_config as _domain_config
from openstack.tests.functional import base
def test_domain_config(self):
    domain_config = self.operator_cloud.identity.create_domain_config(self.domain, identity={'driver': uuid.uuid4().hex}, ldap={'url': uuid.uuid4().hex})
    self.assertIsInstance(domain_config, _domain_config.DomainConfig)
    ldap_url = uuid.uuid4().hex
    domain_config = self.operator_cloud.identity.update_domain_config(self.domain, ldap={'url': ldap_url})
    self.assertIsInstance(domain_config, _domain_config.DomainConfig)
    domain_config = self.operator_cloud.identity.get_domain_config(self.domain)
    self.assertIsInstance(domain_config, _domain_config.DomainConfig)
    self.assertEqual(ldap_url, domain_config.ldap.url)
    result = self.operator_cloud.identity.delete_domain_config(self.domain, ignore_missing=False)
    self.assertIsNone(result)