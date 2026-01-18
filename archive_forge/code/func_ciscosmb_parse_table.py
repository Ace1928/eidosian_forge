from __future__ import absolute_import, division, print_function
import json
import re
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.basic import env_fallback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list, ComplexList
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.community.ciscosmb.plugins.module_utils.ciscosmb_canonical_map import base_interfaces
def ciscosmb_parse_table(table, allow_overflow=True, allow_empty_fields=None):
    if allow_empty_fields is None:
        allow_empty_fields = list()
    fields_end = __get_table_columns_end(table['header'])
    data = __get_table_data(table['data'], fields_end, allow_overflow, allow_empty_fields)
    return data