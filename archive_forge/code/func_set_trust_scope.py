import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
def set_trust_scope(self, id=None, impersonation=False, trustee_user_id=None, trustor_user_id=None):
    self.trust_id = id or uuid.uuid4().hex
    self.trust_impersonation = impersonation
    self.trustee_user_id = trustee_user_id or uuid.uuid4().hex
    self.trustor_user_id = trustor_user_id or uuid.uuid4().hex