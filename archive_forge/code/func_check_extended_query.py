from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.routeros.plugins.module_utils.quoting import (
from ansible_collections.community.routeros.plugins.module_utils.api import (
import re
def check_extended_query(self):
    if self.extended_query['where']:
        for i in self.extended_query['where']:
            if i['or'] is not None:
                if len(i['or']) < 2:
                    self.errors("invalid syntax 'extended_query':'where':'or':%s 'or' requires minimum two items" % i['or'])
                for orv in i['or']:
                    self.check_extended_query_syntax(orv, ":'or':")
            else:
                self.check_extended_query_syntax(i)