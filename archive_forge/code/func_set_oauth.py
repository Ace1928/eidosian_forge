import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
def set_oauth(self, access_token_id=None, consumer_id=None):
    self.oauth_access_token_id = access_token_id or uuid.uuid4().hex
    self.oauth_consumer_id = consumer_id or uuid.uuid4().hex