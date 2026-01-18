import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
def set_application_credential(self, application_credential_id, access_rules=None):
    self.application_credential_id = application_credential_id
    if access_rules is not None:
        self.application_credential_access_rules = access_rules