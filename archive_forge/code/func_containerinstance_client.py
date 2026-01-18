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
@property
def containerinstance_client(self):
    self.log('Getting container instance mgmt client')
    if not self._containerinstance_client:
        self._containerinstance_client = self.get_mgmt_svc_client(ContainerInstanceManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2018-06-01')
    return self._containerinstance_client