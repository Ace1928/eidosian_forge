from keystoneauth1 import exceptions
from keystoneauth1 import identity
from keystoneauth1 import loading
def load_from_options(self, **kwargs):
    if not kwargs.get('oauth2_endpoint'):
        m = 'You must provide an OAuth2.0 Mutual-TLS endpoint.'
        raise exceptions.OptionError(m)
    if not kwargs.get('oauth2_client_id'):
        m = 'You must provide an client credential ID for OAuth2.0 Mutual-TLS Authorization.'
        raise exceptions.OptionError(m)
    return super(OAuth2mTlsClientCredential, self).load_from_options(**kwargs)