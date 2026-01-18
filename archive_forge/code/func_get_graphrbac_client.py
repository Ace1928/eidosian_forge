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
def get_graphrbac_client(self, tenant_id):
    cred = self.azure_auth.azure_credentials
    base_url = self.azure_auth._cloud_environment.endpoints.active_directory_graph_resource_id
    client = GraphRbacManagementClient(cred, tenant_id, base_url)
    return client