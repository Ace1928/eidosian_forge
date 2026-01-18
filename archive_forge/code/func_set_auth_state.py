import abc
import base64
import functools
import hashlib
import json
import threading
from keystoneauth1 import _utils as utils
from keystoneauth1 import access
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
def set_auth_state(self, data):
    """Install existing authentication state for a plugin.

        Take the output of get_auth_state and install that authentication state
        into the current authentication plugin.
        """
    if data:
        auth_data = json.loads(data)
        self.auth_ref = access.create(body=auth_data['body'], auth_token=auth_data['auth_token'])
    else:
        self.auth_ref = None