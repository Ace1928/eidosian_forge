import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
def set_system_scope(self):
    self.system = {'all': True}