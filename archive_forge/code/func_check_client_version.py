from __future__ import absolute_import, division, print_function
import os
import re
import types
import copy
import inspect
import traceback
import json
from os.path import expanduser
from ansible.module_utils.basic import \
from ansible.module_utils.six.moves import configparser
import ansible.module_utils.six.moves.urllib.parse as urlparse
from base64 import b64encode, b64decode
from hashlib import sha256
from hmac import HMAC
from time import time
def check_client_version(self, client_type):
    package_version = AZURE_PKG_VERSIONS.get(client_type.__name__, None)
    if package_version is not None:
        client_name = package_version.get('package_name')
        try:
            client_module = importlib.import_module(client_type.__module__)
            client_version = client_module.VERSION
        except (RuntimeError, AttributeError):
            return
        expected_version = package_version.get('expected_version')
        if Version(client_version) < Version(expected_version):
            self.fail('Installed azure-mgmt-{0} client version is {1}. The minimum supported version is {2}. Try `pip install ansible[azure]`'.format(client_name, client_version, expected_version))
        if Version(client_version) != Version(expected_version):
            self.module.warn('Installed azure-mgmt-{0} client version is {1}. The expected version is {2}. Try `pip install ansible[azure]`'.format(client_name, client_version, expected_version))