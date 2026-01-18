import importlib.metadata
import logging
import warnings
from debtcollector import removals
from debtcollector import renames
from keystoneauth1 import adapter
from oslo_serialization import jsonutils
import packaging.version
import requests
from keystoneclient import _discover
from keystoneclient import access
from keystoneclient.auth import base
from keystoneclient import baseclient
from keystoneclient import exceptions
from keystoneclient.i18n import _
from keystoneclient import session as client_session
def store_auth_ref_into_keyring(self, keyring_key):
    """Store auth_ref into keyring."""
    if self.use_keyring:
        try:
            keyring.set_password('keystoneclient_auth', keyring_key, pickle.dumps(self.auth_ref))
        except Exception as e:
            _logger.warning('Failed to store token into keyring %s', e)