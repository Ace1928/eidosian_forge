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
@unit.skip_if_no_multiple_domains_support
def test_delete_domain_call_db_time(self):
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    domain['enabled'] = False
    PROVIDERS.resource_api.update_domain(domain['id'], domain)
    domain_ref = PROVIDERS.resource_api.get_project(domain['id'])
    with mock.patch.object(resource_sql.Resource, 'get_project') as mock_get_project:
        mock_get_project.return_value = domain_ref
        PROVIDERS.resource_api.delete_domain(domain['id'])
        self.assertEqual(mock_get_project.call_count, 1)