import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@property
def oauth2_credential(self):
    return self.root.get('oauth2_credential')