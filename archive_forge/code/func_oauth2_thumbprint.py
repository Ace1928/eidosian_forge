import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@oauth2_thumbprint.setter
def oauth2_thumbprint(self, value):
    self.root.setdefault('oauth2_credential', {})['x5t#S256'] = value