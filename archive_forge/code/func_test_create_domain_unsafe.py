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
def test_create_domain_unsafe(self):
    """Call ``POST /domains with unsafe names``."""
    unsafe_name = 'i am not / safe'
    self.config_fixture.config(group='resource', domain_name_url_safe='off')
    ref = unit.new_domain_ref(name=unsafe_name)
    self.post('/domains', body={'domain': ref})
    for config_setting in ['new', 'strict']:
        self.config_fixture.config(group='resource', domain_name_url_safe=config_setting)
        ref = unit.new_domain_ref(name=unsafe_name)
        self.post('/domains', body={'domain': ref}, expected_status=http.client.BAD_REQUEST)