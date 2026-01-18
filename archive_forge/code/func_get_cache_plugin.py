from __future__ import (absolute_import, division, print_function)
import hashlib
import os
import string
from collections.abc import Mapping
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.inventory.group import to_safe_group_name as original_safe
from ansible.parsing.utils.addresses import parse_address
from ansible.plugins import AnsiblePlugin
from ansible.plugins.cache import CachePluginAdjudicator as CacheObject
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import string_types
from ansible.template import Templar
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars, load_extra_vars
def get_cache_plugin(plugin_name, **kwargs):
    try:
        cache = CacheObject(plugin_name, **kwargs)
    except AnsibleError as e:
        if 'fact_caching_connection' in to_native(e):
            raise AnsibleError("error, '%s' inventory cache plugin requires the one of the following to be set to a writeable directory path:\nansible.cfg:\n[default]: fact_caching_connection,\n[inventory]: cache_connection;\nEnvironment:\nANSIBLE_INVENTORY_CACHE_CONNECTION,\nANSIBLE_CACHE_PLUGIN_CONNECTION." % plugin_name)
        else:
            raise e
    if plugin_name != 'memory' and kwargs and (not getattr(cache._plugin, '_options', None)):
        raise AnsibleError('Unable to use cache plugin {0} for inventory. Cache options were provided but may not reconcile correctly unless set via set_options. Refer to the porting guide if the plugin derives user settings from ansible.constants.'.format(plugin_name))
    return cache