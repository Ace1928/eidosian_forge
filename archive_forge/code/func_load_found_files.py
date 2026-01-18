from __future__ import (absolute_import, division, print_function)
import os
from ansible.errors import AnsibleParserError
from ansible.module_utils.common.text.converters import to_native
from ansible.plugins.vars import BaseVarsPlugin
from ansible.utils.path import basedir
from ansible.inventory.group import InventoryObjectType
from ansible.utils.vars import combine_vars
def load_found_files(self, loader, data, found_files):
    for found in found_files:
        new_data = loader.load_from_file(found, cache=True, unsafe=True)
        if new_data:
            data = combine_vars(data, new_data)
    return data