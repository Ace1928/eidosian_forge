from __future__ import (absolute_import, division, print_function)
import os
import re
import traceback
from operator import itemgetter
def search_host(self, search_string):
    results = []
    for host_entry in self.config_data:
        if host_entry.get('type') != 'entry':
            continue
        if host_entry.get('host') == '*':
            continue
        searchable_information = host_entry.get('host')
        for key, value in host_entry.get('options').items():
            if isinstance(value, list):
                value = ' '.join(value)
            if isinstance(value, int):
                value = str(value)
            searchable_information += ' ' + value
        if search_string in searchable_information:
            results.append(host_entry)
    return results