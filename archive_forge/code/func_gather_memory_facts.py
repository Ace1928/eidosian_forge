from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.frr.frr.plugins.module_utils.network.frr.frr import (
def gather_memory_facts(self, data):
    mem_details = data.split('\n\n')
    mem_stats = {}
    mem_counters = {'total_heap_allocated': 'Total heap allocated:(?:\\s*)(.*)', 'holding_block_headers': 'Holding block headers:(?:\\s*)(.*)', 'used_small_blocks': 'Used small blocks:(?:\\s*)(.*)', 'used_ordinary_blocks': 'Used ordinary blocks:(?:\\s*)(.*)', 'free_small_blocks': 'Free small blocks:(?:\\s*)(.*)', 'free_ordinary_blocks': 'Free ordinary blocks:(?:\\s*)(.*)', 'ordinary_blocks': 'Ordinary blocks:(?:\\s*)(.*)', 'small_blocks': 'Small blocks:(?:\\s*)(.*)', 'holding_blocks': 'Holding blocks:(?:\\s*)(.*)'}
    for item in mem_details:
        daemon = self._parse_daemons(item)
        mem_stats[daemon] = {}
        for fact, pattern in iteritems(mem_counters):
            mem_stats[daemon][fact] = self.parse_facts(pattern, item)
    return mem_stats