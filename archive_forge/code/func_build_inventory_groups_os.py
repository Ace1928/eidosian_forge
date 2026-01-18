from __future__ import (absolute_import, division, print_function)
import json
import re
import time
import os
from ansible.plugins.inventory import BaseInventoryPlugin
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible.module_utils.six import raise_from
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.community.general.plugins.module_utils.lxd import LXDClient, LXDClientException
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def build_inventory_groups_os(self, group_name):
    """create group by attribute: os

        Args:
            str(group_name): Group name
        Kwargs:
            None
        Raises:
            None
        Returns:
            None"""
    if group_name not in self.inventory.groups:
        self.inventory.add_group(group_name)
    gen_instances = [instance_name for instance_name in self.inventory.hosts if 'ansible_lxd_os' in self.inventory.get_host(instance_name).get_vars()]
    for instance_name in gen_instances:
        if self.groupby[group_name].get('attribute').lower() == self.inventory.get_host(instance_name).get_vars().get('ansible_lxd_os'):
            self.inventory.add_child(group_name, instance_name)