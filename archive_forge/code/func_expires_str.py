import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@expires_str.setter
def expires_str(self, value):
    self._token['expires'] = value