import uuid
from oslo_utils import timeutils
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3.contrib import trusts
def test_create_imp(self):
    ref = self.new_ref()
    ref['trustor_user_id'] = uuid.uuid4().hex
    ref['trustee_user_id'] = uuid.uuid4().hex
    ref['impersonation'] = True
    super(TrustTests, self).test_create(ref=ref)