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
def parse_resource_to_dict(self, resource):
    """
        Return a dict of the give resource, which contains name and resource group.

        :param resource: It can be a resource name, id or a dict contains name and resource group.
        """
    resource_dict = parse_resource_id(resource) if not isinstance(resource, dict) else resource
    resource_dict['resource_group'] = resource_dict.get('resource_group', self.resource_group)
    resource_dict['subscription_id'] = resource_dict.get('subscription_id', self.subscription_id)
    return resource_dict