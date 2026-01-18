from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.routeros.plugins.module_utils.quoting import (
from ansible_collections.community.routeros.plugins.module_utils.api import (
import re
def api_extended_query(self):
    self.query_keys = {}
    for k in self.extended_query['attributes']:
        if k == 'id':
            self.errors("'extended_query':'attributes':'%s' must be '.id'" % k)
        self.query_keys[k] = Key(k)
    try:
        if self.extended_query['where']:
            where_args = []
            for i in self.extended_query['where']:
                if i['or']:
                    where_or_args = []
                    for ior in i['or']:
                        where_or_args.append(self.build_api_extended_query(ior))
                    where_args.append(Or(*where_or_args))
                else:
                    where_args.append(self.build_api_extended_query(i))
            select = self.api_path.select(*self.query_keys).where(*where_args)
        else:
            select = self.api_path.select(*self.extended_query['attributes'])
        for row in select:
            self.result['message'].append(row)
        self.return_result(False)
    except LibRouterosError as e:
        self.errors(e)