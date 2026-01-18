from __future__ import (absolute_import, division, print_function)
import os
import re
from collections.abc import MutableMapping, MutableSequence
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils import six
from ansible.plugins.loader import connection_loader
from ansible.utils.display import Display
def clean_facts(facts):
    """ remove facts that can override internal keys or otherwise deemed unsafe """
    data = module_response_deepcopy(facts)
    remove_keys = set()
    fact_keys = set(data.keys())
    for magic_var in C.MAGIC_VARIABLE_MAPPING:
        remove_keys.update(fact_keys.intersection(C.MAGIC_VARIABLE_MAPPING[magic_var]))
    remove_keys.update(fact_keys.intersection(C.COMMON_CONNECTION_VARS))
    for conn_path in connection_loader.all(path_only=True):
        conn_name = os.path.splitext(os.path.basename(conn_path))[0]
        re_key = re.compile('^ansible_%s_' % re.escape(conn_name))
        for fact_key in fact_keys:
            if re_key.match(fact_key) and (not fact_key.endswith(('_bridge', '_gwbridge'))) or fact_key.startswith('ansible_become_'):
                remove_keys.add(fact_key)
    for hard in C.RESTRICTED_RESULT_KEYS + C.INTERNAL_RESULT_KEYS:
        if hard in fact_keys:
            remove_keys.add(hard)
    re_interp = re.compile('^ansible_.*_interpreter$')
    for fact_key in fact_keys:
        if re_interp.match(fact_key):
            remove_keys.add(fact_key)
    for r_key in remove_keys:
        if not r_key.startswith('ansible_ssh_host_key_'):
            display.warning('Removed restricted key from module data: %s' % r_key)
            del data[r_key]
    return strip_internal_keys(data)