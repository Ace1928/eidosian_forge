import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@issued.setter
def issued(self, value):
    self.issued_str = value.isoformat()