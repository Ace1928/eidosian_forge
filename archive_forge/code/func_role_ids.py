import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@property
def role_ids(self):
    return [r['id'] for r in self.root.get('roles', [])]