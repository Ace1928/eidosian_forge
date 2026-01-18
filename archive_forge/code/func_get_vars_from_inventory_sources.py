from __future__ import annotations
import os
from functools import lru_cache
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.inventory.group import InventoryObjectType
from ansible.plugins.loader import vars_loader
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars
def get_vars_from_inventory_sources(loader, sources, entities, stage):
    data = {}
    for path in sources:
        if path is None:
            continue
        if ',' in path and (not os.path.exists(path)):
            continue
        elif not os.path.isdir(path):
            path = os.path.dirname(path)
        if (new_vars := get_vars_from_path(loader, path, entities, stage)) != {}:
            data = combine_vars(data, new_vars)
    return data