import uuid
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import tokenless_auth
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_domain_id_scope_header_pass(self):
    domain_id = uuid.uuid4().hex
    auth, session = self.create(auth_url=self.TEST_URL, domain_id=domain_id)
    session.get(self.TEST_URL, authenticated=True)
    self.assertRequestHeaderEqual('X-Domain-Id', domain_id)