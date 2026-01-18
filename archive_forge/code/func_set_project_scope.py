import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
def set_project_scope(self, id=None, name=None, domain_id=None, domain_name=None, is_domain=None):
    self.project_id = id or uuid.uuid4().hex
    self.project_name = name or uuid.uuid4().hex
    self.project_domain_id = domain_id or uuid.uuid4().hex
    self.project_domain_name = domain_name or uuid.uuid4().hex
    if is_domain is not None:
        self.project_is_domain = is_domain