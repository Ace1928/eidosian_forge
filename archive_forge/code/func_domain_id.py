import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@domain_id.setter
def domain_id(self, value):
    self.root.setdefault('domain', {})['id'] = value