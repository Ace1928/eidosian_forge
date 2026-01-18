from keystoneauth1 import discover
from keystoneauth1.identity.generic import base
from keystoneauth1.identity import v2
from keystoneauth1.identity import v3
def get_cache_id_elements(self):
    elements = super(Password, self).get_cache_id_elements(_implemented=True)
    elements['username'] = self._username
    elements['user_id'] = self._user_id
    elements['password'] = self._password
    elements['user_domain_id'] = self.user_domain_id
    elements['user_domain_name'] = self.user_domain_name
    return elements