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
def test_list_project_is_domain_filter(self):
    """Call ``GET /projects?is_domain=True/False``."""
    r = self.get('/projects?is_domain=True', expected_status=200)
    initial_number_is_domain_true = len(r.result['projects'])
    r = self.get('/projects?is_domain=False', expected_status=200)
    initial_number_is_domain_false = len(r.result['projects'])
    new_is_domain_project = unit.new_project_ref(is_domain=True)
    new_is_domain_project = PROVIDERS.resource_api.create_project(new_is_domain_project['id'], new_is_domain_project)
    new_is_domain_project2 = unit.new_project_ref(is_domain=True)
    new_is_domain_project2 = PROVIDERS.resource_api.create_project(new_is_domain_project2['id'], new_is_domain_project2)
    number_is_domain_true = initial_number_is_domain_true + 2
    r = self.get('/projects?is_domain=True', expected_status=200)
    self.assertThat(r.result['projects'], matchers.HasLength(number_is_domain_true))
    self.assertIn(new_is_domain_project['id'], [p['id'] for p in r.result['projects']])
    self.assertIn(new_is_domain_project2['id'], [p['id'] for p in r.result['projects']])
    new_regular_project = unit.new_project_ref(domain_id=self.domain_id)
    new_regular_project = PROVIDERS.resource_api.create_project(new_regular_project['id'], new_regular_project)
    number_is_domain_false = initial_number_is_domain_false + 1
    r = self.get('/projects?is_domain=True', expected_status=200)
    self.assertThat(r.result['projects'], matchers.HasLength(number_is_domain_true))
    r = self.get('/projects?is_domain=False', expected_status=200)
    self.assertThat(r.result['projects'], matchers.HasLength(number_is_domain_false))
    self.assertIn(new_regular_project['id'], [p['id'] for p in r.result['projects']])