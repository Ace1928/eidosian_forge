import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@expires.setter
def expires(self, value):
    self.expires_str = value.isoformat()