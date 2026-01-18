import uuid
from oslo_utils import timeutils
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3.contrib import trusts
def test_create_role_id_and_names(self):
    ref = self.new_ref()
    ref['trustor_user_id'] = uuid.uuid4().hex
    ref['trustee_user_id'] = uuid.uuid4().hex
    ref['impersonation'] = False
    req_ref = ref.copy()
    req_ref.pop('id')
    ref['role_names'] = ['atestrole']
    ref['role_ids'] = [uuid.uuid4().hex]
    req_ref['roles'] = [{'name': 'atestrole'}, {'id': ref['role_ids'][0]}]
    super(TrustTests, self).test_create(ref=ref, req_ref=req_ref)