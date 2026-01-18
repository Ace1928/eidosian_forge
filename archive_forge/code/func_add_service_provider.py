import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
def add_service_provider(self, sp_id, sp_auth_url, sp_url):
    _service_providers = self.root.setdefault('service_providers', [])
    sp = {'id': sp_id, 'auth_url': sp_auth_url, 'sp_url': sp_url}
    _service_providers.append(sp)
    return sp