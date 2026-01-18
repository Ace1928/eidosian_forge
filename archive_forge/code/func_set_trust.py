import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
def set_trust(self, id=None, trustee_user_id=None):
    self.trust_id = id or uuid.uuid4().hex
    self.trustee_user_id = trustee_user_id or uuid.uuid4().hex