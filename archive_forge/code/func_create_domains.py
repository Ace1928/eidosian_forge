import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def create_domains(count):
    for _ in range(count):
        domain = unit.new_domain_ref()
        self.domain_list.append(PROVIDERS.resource_api.create_domain(domain['id'], domain))