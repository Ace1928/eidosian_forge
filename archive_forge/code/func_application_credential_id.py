import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@application_credential_id.setter
def application_credential_id(self, value):
    application_credential = self.root.setdefault('application_credential', {})
    application_credential.setdefault('id', value)