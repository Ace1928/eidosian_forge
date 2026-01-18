import abc
import urllib.parse
from keystoneauth1 import _utils as utils
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1.identity import base
def get_auth_ref(self, session, **kwargs):
    if not self._plugin:
        self._plugin = self._do_create_plugin(session)
    return self._plugin.get_auth_ref(session, **kwargs)