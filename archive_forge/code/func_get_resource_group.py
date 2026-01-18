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
def get_resource_group(self, resource_group):
    """
        Fetch a resource group.

        :param resource_group: name of a resource group
        :return: resource group object
        """
    try:
        return self.rm_client.resource_groups.get(resource_group)
    except Exception as exc:
        self.fail('Error retrieving resource group {0} - {1}'.format(resource_group, str(exc)))