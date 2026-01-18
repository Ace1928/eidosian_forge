from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.dict_transformations import _camel_to_snake
from ansible.module_utils._text import to_native
from datetime import datetime, timedelta
def list_all_items(self):
    self.log('List all items in subscription')
    try:
        if self.type != 'namespace':
            return []
        response = self.servicebus_client.namespaces.list()
        return [x for x in response if self.has_tags(x.tags, self.tags)]
    except Exception as exc:
        self.fail('Failed to list all items - {0}'.format(str(exc)))
    return []