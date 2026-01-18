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
def test_forbid_operations_on_federated_domain(self):
    """Make sure one cannot operate on federated domain.

        This includes operations like create, update, delete
        on domain identified by id and name where difference variations of
        id 'Federated' are used.

        """

    def create_domains():
        for variation in ('Federated', 'FEDERATED', 'federated', 'fEderated'):
            domain = unit.new_domain_ref()
            domain['id'] = variation
            yield domain
    for domain in create_domains():
        self.assertRaises(AssertionError, PROVIDERS.resource_api.create_domain, domain['id'], domain)
        self.assertRaises(AssertionError, PROVIDERS.resource_api.update_domain, domain['id'], domain)
        self.assertRaises(exception.DomainNotFound, PROVIDERS.resource_api.delete_domain, domain['id'])
        domain['id'], domain['name'] = (domain['name'], domain['id'])
        self.assertRaises(AssertionError, PROVIDERS.resource_api.create_domain, domain['id'], domain)
        self.assertRaises(AssertionError, PROVIDERS.resource_api.update_domain, domain['id'], domain)
        self.assertRaises(exception.DomainNotFound, PROVIDERS.resource_api.delete_domain, domain['id'])