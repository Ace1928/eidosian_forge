import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@oauth_access_token_id.setter
def oauth_access_token_id(self, value):
    self.root.setdefault('OS-OAUTH1', {})['access_token_id'] = value