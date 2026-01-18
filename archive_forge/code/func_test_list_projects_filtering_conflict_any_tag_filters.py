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
def test_list_projects_filtering_conflict_any_tag_filters(self):
    """Call ``GET /projects?tags-any={tags}&not-tags-any={tags}``."""
    project, tags = self._create_project_and_tags(num_of_tags=2)
    tag_string = ','.join(tags)
    url = '/projects?tags-any=%(values)s&not-tags-any=%(values)s'
    resp = self.get(url % {'values': tag_string})
    self.assertValidProjectListResponse(resp)
    self.assertEqual(len(resp.result['projects']), 0)