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
def load_cache_plugin(self):
    plugin_name = self.get_option('cache_plugin')
    cache_option_keys = [('_uri', 'cache_connection'), ('_timeout', 'cache_timeout'), ('_prefix', 'cache_prefix')]
    cache_options = dict(((opt[0], self.get_option(opt[1])) for opt in cache_option_keys if self.get_option(opt[1]) is not None))
    self._cache = get_cache_plugin(plugin_name, **cache_options)