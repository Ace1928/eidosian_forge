import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import grant as gp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_can_revoke_grant_from_group_on_domain(self):
    group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=CONF.identity.default_domain_id))
    domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
    PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, group_id=group['id'], domain_id=domain['id'])
    with self.test_client() as c:
        c.delete('/v3/domains/%s/groups/%s/roles/%s' % (domain['id'], group['id'], self.bootstrapper.reader_role_id), headers=self.headers)