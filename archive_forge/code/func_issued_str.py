import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@issued_str.setter
def issued_str(self, value):
    self._token['issued_at'] = value