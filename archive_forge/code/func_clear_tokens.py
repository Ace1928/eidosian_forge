import uuid
import fixtures
from keystoneauth1.fixture import v2
from keystoneauth1.fixture import v3
import os_service_types
def clear_tokens(self):
    self.v2_token = v2.Token(tenant_id=self.project_id)
    self.v3_token = v3.Token(project_id=self.project_id)