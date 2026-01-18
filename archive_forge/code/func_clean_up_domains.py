import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def clean_up_domains():
    for domain in self.domain_list:
        PROVIDERS.resource_api.update_domain(domain['id'], {'enabled': False})
        PROVIDERS.resource_api.delete_domain(domain['id'])