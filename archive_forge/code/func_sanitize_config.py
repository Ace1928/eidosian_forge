from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import ConnectionError
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.vyos import (
def sanitize_config(config, result):
    result['filtered'] = list()
    index_to_filter = list()
    for regex in CONFIG_FILTERS:
        for index, line in enumerate(list(config)):
            if regex.search(line):
                result['filtered'].append(line)
                index_to_filter.append(index)
    for filter_index in sorted(index_to_filter, reverse=True):
        del config[filter_index]