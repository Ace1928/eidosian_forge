import uuid
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_domain_duplicate_conflict_gives_name(self):
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    domain['id'] = uuid.uuid4().hex
    try:
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
    except exception.Conflict as e:
        self.assertIn('%s' % domain['name'], repr(e))
    else:
        self.fail('Creating duplicate domain did not raise a conflict')