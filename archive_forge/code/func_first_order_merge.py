from __future__ import (absolute_import, division, print_function)
from collections.abc import MutableMapping
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.plugins.loader import cache_loader
from ansible.utils.display import Display
def first_order_merge(self, key, value):
    host_facts = {key: value}
    try:
        host_cache = self._plugin.get(key)
        if host_cache:
            host_cache.update(value)
            host_facts[key] = host_cache
    except KeyError:
        pass
    super(FactCache, self).update(host_facts)