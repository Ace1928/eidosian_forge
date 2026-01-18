from __future__ import absolute_import, division, print_function
import json
import re
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.basic import env_fallback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list, ComplexList
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.community.ciscosmb.plugins.module_utils.ciscosmb_canonical_map import base_interfaces
def ciscosmb_split_to_tables(data):
    TABLE_HEADER = re.compile('^---+ +-+.*$')
    EMPTY_LINE = re.compile('^ *$')
    tables = dict()
    tableno = -1
    lineno = 0
    tabledataget = False
    for line in data.splitlines():
        if re.match(EMPTY_LINE, line):
            tabledataget = False
            continue
        if re.match(TABLE_HEADER, line):
            tableno += 1
            tabledataget = True
            lineno = 0
            tables[tableno] = dict()
            tables[tableno]['header'] = line
            tables[tableno]['data'] = dict()
            continue
        if tabledataget:
            tables[tableno]['data'][lineno] = line
            lineno += 1
            continue
    return tables