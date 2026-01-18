import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@oauth_consumer_id.setter
def oauth_consumer_id(self, value):
    self.root.setdefault('OS-OAUTH1', {})['consumer_id'] = value