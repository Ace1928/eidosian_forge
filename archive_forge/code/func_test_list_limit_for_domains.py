import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
@unit.skip_if_no_multiple_domains_support
def test_list_limit_for_domains(self):

    def create_domains(count):
        for _ in range(count):
            domain = unit.new_domain_ref()
            self.domain_list.append(PROVIDERS.resource_api.create_domain(domain['id'], domain))

    def clean_up_domains():
        for domain in self.domain_list:
            PROVIDERS.resource_api.update_domain(domain['id'], {'enabled': False})
            PROVIDERS.resource_api.delete_domain(domain['id'])
    self.domain_list = []
    create_domains(6)
    self.addCleanup(clean_up_domains)
    for x in range(1, 7):
        self.config_fixture.config(group='resource', list_limit=x)
        hints = driver_hints.Hints()
        entities = PROVIDERS.resource_api.list_domains(hints=hints)
        self.assertThat(entities, matchers.HasLength(hints.limit['limit']))