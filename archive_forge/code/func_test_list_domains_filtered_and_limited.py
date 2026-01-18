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
def test_list_domains_filtered_and_limited(self):

    def create_domains(domain_count, domain_name_prefix):
        for _ in range(domain_count):
            domain_name = '%s-%s' % (domain_name_prefix, uuid.uuid4().hex)
            domain = unit.new_domain_ref(name=domain_name)
            self.domain_list[domain_name] = PROVIDERS.resource_api.create_domain(domain['id'], domain)

    def clean_up_domains():
        for _, domain in self.domain_list.items():
            domain['enabled'] = False
            PROVIDERS.resource_api.update_domain(domain['id'], domain)
            PROVIDERS.resource_api.delete_domain(domain['id'])
    self.domain_list = {}
    create_domains(2, 'domaingroup1')
    create_domains(3, 'domaingroup2')
    self.addCleanup(clean_up_domains)
    unfiltered_domains = PROVIDERS.resource_api.list_domains()
    self.config_fixture.config(list_limit=4)
    hints = driver_hints.Hints()
    entities = PROVIDERS.resource_api.list_domains(hints=hints)
    self.assertThat(entities, matchers.HasLength(hints.limit['limit']))
    self.assertTrue(hints.limit['truncated'])
    hints = driver_hints.Hints()
    hints.add_filter('name', unfiltered_domains[3]['name'])
    entities = PROVIDERS.resource_api.list_domains(hints=hints)
    self.assertThat(entities, matchers.HasLength(1))
    self.assertEqual(entities[0], unfiltered_domains[3])
    hints = driver_hints.Hints()
    hints.add_filter('name', 'domaingroup1', comparator='startswith')
    entities = PROVIDERS.resource_api.list_domains(hints=hints)
    self.assertThat(entities, matchers.HasLength(2))
    self.assertThat(entities[0]['name'], matchers.StartsWith('domaingroup1'))
    self.assertThat(entities[1]['name'], matchers.StartsWith('domaingroup1'))