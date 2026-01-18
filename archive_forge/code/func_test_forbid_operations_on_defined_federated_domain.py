import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
from keystone.tests.unit import utils as test_utils
def test_forbid_operations_on_defined_federated_domain(self):
    """Make sure one cannot operate on a user-defined federated domain.

        This includes operations like create, update, delete.

        """
    non_default_name = 'beta_federated_domain'
    self.config_fixture.config(group='federation', federated_domain_name=non_default_name)
    domain = unit.new_domain_ref(name=non_default_name)
    self.assertRaises(AssertionError, PROVIDERS.resource_api.create_domain, domain['id'], domain)
    self.assertRaises(exception.DomainNotFound, PROVIDERS.resource_api.delete_domain, domain['id'])
    self.assertRaises(AssertionError, PROVIDERS.resource_api.update_domain, domain['id'], domain)