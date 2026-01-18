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
def get_subnet_detail(self, subnet_id):
    vnet_detail = subnet_id.split('/Microsoft.Network/virtualNetworks/')[1].split('/subnets/')
    return dict(resource_group=subnet_id.split('resourceGroups/')[1].split('/')[0], vnet_name=vnet_detail[0], subnet_name=vnet_detail[1])