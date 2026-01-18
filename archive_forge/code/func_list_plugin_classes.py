from __future__ import (absolute_import, division, print_function)
import os
from ansible import context
from ansible import constants as C
from ansible.collections.list import list_collections
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible.plugins import loader
from ansible.utils.display import Display
from ansible.utils.collection_loader._collection_finder import _get_collection_path
def list_plugin_classes(ptype, collection=None):
    plugins = list_plugins(ptype, collection)
    return [plugins[k][1] for k in plugins.keys()]