import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@tenant_name.setter
def tenant_name(self, value):
    self._token.setdefault('tenant', {})['name'] = value