from __future__ import (absolute_import, division, print_function)
import fnmatch
import os
import sys
import re
import itertools
import traceback
from operator import attrgetter
from random import shuffle
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleOptionsError, AnsibleParserError
from ansible.inventory.data import InventoryData
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.parsing.utils.addresses import parse_address
from ansible.plugins.loader import inventory_loader
from ansible.utils.helpers import deduplicate_list
from ansible.utils.path import unfrackpath
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars
from ansible.vars.plugins import get_vars_from_inventory_sources
def get_hosts(self, pattern='all', ignore_limits=False, ignore_restrictions=False, order=None):
    """
        Takes a pattern or list of patterns and returns a list of matching
        inventory host names, taking into account any active restrictions
        or applied subsets
        """
    hosts = []
    if isinstance(pattern, list):
        pattern_list = pattern[:]
    else:
        pattern_list = [pattern]
    if pattern_list:
        if not ignore_limits and self._subset:
            pattern_list.extend(self._subset)
        if not ignore_restrictions and self._restriction:
            pattern_list.extend(self._restriction)
        pattern_hash = tuple(pattern_list)
        if pattern_hash not in self._hosts_patterns_cache:
            patterns = split_host_pattern(pattern)
            hosts = self._evaluate_patterns(patterns)
            if not ignore_limits and self._subset:
                subset_uuids = set((s._uuid for s in self._evaluate_patterns(self._subset)))
                hosts = [h for h in hosts if h._uuid in subset_uuids]
            if not ignore_restrictions and self._restriction:
                hosts = [h for h in hosts if h.name in self._restriction]
            self._hosts_patterns_cache[pattern_hash] = deduplicate_list(hosts)
        if order in ['sorted', 'reverse_sorted']:
            hosts = sorted(self._hosts_patterns_cache[pattern_hash][:], key=attrgetter('name'), reverse=order == 'reverse_sorted')
        elif order == 'reverse_inventory':
            hosts = self._hosts_patterns_cache[pattern_hash][::-1]
        else:
            hosts = self._hosts_patterns_cache[pattern_hash][:]
            if order == 'shuffle':
                shuffle(hosts)
            elif order not in [None, 'inventory']:
                raise AnsibleOptionsError("Invalid 'order' specified for inventory hosts: %s" % order)
    return hosts