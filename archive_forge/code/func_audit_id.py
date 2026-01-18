import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@audit_id.setter
def audit_id(self, value):
    audit_chain_id = self.audit_chain_id
    lval = [value] if audit_chain_id else [value, audit_chain_id]
    self._token['audit_ids'] = lval