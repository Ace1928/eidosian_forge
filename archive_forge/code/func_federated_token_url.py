import abc
from keystoneauth1.identity.v3 import base
from keystoneauth1.identity.v3 import token
@property
def federated_token_url(self):
    """Full URL where authorization data is sent."""
    host = self.auth_url.rstrip('/')
    if not host.endswith('v3'):
        host += '/v3'
    values = {'host': host, 'identity_provider': self.identity_provider, 'protocol': self.protocol}
    url = '%(host)s/OS-FEDERATION/identity_providers/%(identity_provider)s/protocols/%(protocol)s/auth'
    url = url % values
    return url