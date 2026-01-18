import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_get_domain_config(self):
    config = fixtures.DomainConfig(self.client, self.test_domain.id)
    self.useFixture(config)
    config_ret = self.client.domain_configs.get(self.test_domain.id)
    self.check_domain_config(config_ret, config.ref)