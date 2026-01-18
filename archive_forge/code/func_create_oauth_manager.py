from keystoneclient.i18n import _
from keystoneclient.v3.contrib.oauth1 import access_tokens
from keystoneclient.v3.contrib.oauth1 import consumers
from keystoneclient.v3.contrib.oauth1 import request_tokens
def create_oauth_manager(self):
    try:
        import oauthlib
    except ImportError:
        return OAuthManagerOptionalImportProxy()
    else:
        return OAuthManager(self)