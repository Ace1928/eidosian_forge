import contextlib
import os
import uuid
import warnings
import fixtures
from keystoneauth1 import fixture
from keystoneauth1 import identity as ksa_identity
from keystoneauth1 import session as ksa_session
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import testresources
from keystoneclient.auth import identity as ksc_identity
from keystoneclient.common import cms
from keystoneclient import session as ksc_session
from keystoneclient import utils
from keystoneclient.v2_0 import client as v2_client
from keystoneclient.v3 import client as v3_client
def new_client(self):
    t = fixture.V3Token(user_id=self.user_id)
    t.set_project_scope()
    s = t.add_service('identity')
    s.add_standard_endpoints(public=self.TEST_URL, admin=self.TEST_URL)
    d = fixture.V3Discovery(self.TEST_URL)
    headers = {'X-Subject-Token': uuid.uuid4().hex}
    self.requests.register_uri('POST', self.TEST_URL + '/auth/tokens', headers=headers, json=t)
    self.requests.register_uri('GET', self.TEST_URL, json={'version': d})
    a = ksa_identity.V3Password(username=uuid.uuid4().hex, password=uuid.uuid4().hex, user_domain_id=uuid.uuid4().hex, auth_url=self.TEST_URL)
    s = ksa_session.Session(auth=a)
    return v3_client.Client(session=s)