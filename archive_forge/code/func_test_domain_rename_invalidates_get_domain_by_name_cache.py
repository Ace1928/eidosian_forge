import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
import keystone.conf
from keystone import exception
from keystone.resource.backends import sql as resource_sql
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import utils as test_utils
@unit.skip_if_cache_disabled('resource')
@unit.skip_if_no_multiple_domains_support
def test_domain_rename_invalidates_get_domain_by_name_cache(self):
    domain = unit.new_domain_ref()
    domain_id = domain['id']
    domain_name = domain['name']
    PROVIDERS.resource_api.create_domain(domain_id, domain)
    domain_ref = PROVIDERS.resource_api.get_domain_by_name(domain_name)
    domain_ref['name'] = uuid.uuid4().hex
    PROVIDERS.resource_api.update_domain(domain_id, domain_ref)
    self.assertRaises(exception.DomainNotFound, PROVIDERS.resource_api.get_domain_by_name, domain_name)