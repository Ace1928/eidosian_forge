import fixtures
from keystone import auth
import keystone.server
def clear_auth_plugin_registry(self):
    auth.core.AUTH_METHODS.clear()
    auth.core.AUTH_PLUGINS_LOADED = False