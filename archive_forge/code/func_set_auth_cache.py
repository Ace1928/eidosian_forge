import copy
import os.path
import typing as ty
from urllib import parse
import warnings
from keystoneauth1 import discover
import keystoneauth1.exceptions.catalog
from keystoneauth1.loading import adapter as ks_load_adap
from keystoneauth1 import session as ks_session
import os_service_types
import requestsexceptions
from openstack import _log
from openstack.config import _util
from openstack.config import defaults as config_defaults
from openstack import exceptions
from openstack import proxy
from openstack import version as openstack_version
from openstack import warnings as os_warnings
def set_auth_cache(self):
    if self.skip_auth_cache():
        return
    cache_id = self._auth.get_cache_id()
    state = self._auth.get_auth_state()
    try:
        if state:
            keyring.set_password('openstacksdk', cache_id, state)
    except RuntimeError:
        self.log.debug('Failed to set auth into keyring')