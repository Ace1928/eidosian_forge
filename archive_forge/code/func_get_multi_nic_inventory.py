from __future__ import absolute_import, division, print_function
import json
import os
import random
import string
import gzip
from io import BytesIO
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import text_type
from ansible.module_utils.six.moves import http_client
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def get_multi_nic_inventory(self, resource_type):
    ret = True
    entries = []
    if resource_type == 'Systems':
        resource_uris = self.systems_uris
    elif resource_type == 'Manager':
        resource_uris = self.manager_uris
    for resource_uri in resource_uris:
        inventory = self.get_nic_inventory(resource_uri)
        ret = inventory.pop('ret') and ret
        if 'entries' in inventory:
            entries.append(({'resource_uri': resource_uri}, inventory['entries']))
    return dict(ret=ret, entries=entries)