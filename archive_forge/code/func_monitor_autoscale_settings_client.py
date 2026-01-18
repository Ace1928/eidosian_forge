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
def monitor_autoscale_settings_client(self):
    self.log('Getting monitor client for autoscale_settings')
    if not self._monitor_autoscale_settings_client:
        self._monitor_autoscale_settings_client = self.get_mgmt_svc_client(MonitorManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, api_version='2015-04-01', is_track2=True)
    return self._monitor_autoscale_settings_client