from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.routeros.plugins.module_utils.quoting import (
from ansible_collections.community.routeros.plugins.module_utils.api import (
import re
def check_extended_query_syntax(self, test_atr, or_msg=''):
    if test_atr['is'] == 'in' and (not isinstance(test_atr['value'], list)):
        self.errors("invalid syntax 'extended_query':'where':%s%s 'value' must be a type list" % (or_msg, test_atr))