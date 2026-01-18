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
def test_delete_domain_hierarchy(self):
    """Call ``DELETE /domains/{domain_id}``."""
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    root_project = unit.new_project_ref(domain_id=domain['id'])
    root_project = PROVIDERS.resource_api.create_project(root_project['id'], root_project)
    leaf_project = unit.new_project_ref(domain_id=domain['id'], parent_id=root_project['id'])
    PROVIDERS.resource_api.create_project(leaf_project['id'], leaf_project)
    self.patch('/domains/%(domain_id)s' % {'domain_id': domain['id']}, body={'domain': {'enabled': False}})
    self.delete('/domains/%(domain_id)s' % {'domain_id': domain['id']})
    self.assertRaises(exception.DomainNotFound, PROVIDERS.resource_api.get_domain, domain['id'])
    self.assertRaises(exception.ProjectNotFound, PROVIDERS.resource_api.get_project, root_project['id'])
    self.assertRaises(exception.ProjectNotFound, PROVIDERS.resource_api.get_project, leaf_project['id'])