from __future__ import absolute_import, division, print_function
import shlex
import time
import traceback
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible.module_utils.basic import human_to_bytes
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text, to_native
def have_networks_changed(new_networks, old_networks):
    """Special case list checking for networks to sort aliases"""
    if new_networks is None:
        return False
    old_networks = old_networks or []
    if len(new_networks) != len(old_networks):
        return True
    zip_data = zip(sorted(new_networks, key=lambda k: k['id']), sorted(old_networks, key=lambda k: k['id']))
    for new_item, old_item in zip_data:
        new_item = dict(new_item)
        old_item = dict(old_item)
        if 'aliases' in new_item:
            new_item['aliases'] = sorted(new_item['aliases'] or [])
        if 'aliases' in old_item:
            old_item['aliases'] = sorted(old_item['aliases'] or [])
        if has_dict_changed(new_item, old_item):
            return True
    return False