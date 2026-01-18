from __future__ import absolute_import, division, print_function
from copy import deepcopy
import re
import os
import ast
import datetime
import shutil
import tempfile
from ansible.module_utils.basic import json
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves import filterfalse
from ansible.module_utils.six.moves.urllib.parse import urlencode, urljoin
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.connection import Connection
from ansible_collections.cisco.mso.plugins.module_utils.constants import NDO_API_VERSION_PATH_FORMAT
def make_subnets(self, data, is_bd_subnet=True):
    """Create a subnets list from input"""
    if data is None:
        return None
    subnets = []
    for subnet in data:
        if 'subnet' in subnet:
            subnet['ip'] = subnet.get('subnet')
        if subnet.get('description') is None:
            subnet['description'] = subnet.get('subnet')
        subnet_payload = dict(ip=subnet.get('ip'), description=str(subnet.get('description')), scope=subnet.get('scope'), shared=subnet.get('shared'), noDefaultGateway=subnet.get('no_default_gateway'))
        if is_bd_subnet:
            subnet_payload.update(dict(querier=subnet.get('querier'), primary=subnet.get('primary'), virtual=subnet.get('virtual')))
        subnets.append(subnet_payload)
    return subnets