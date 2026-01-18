from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.routeros.plugins.module_utils.api import (
def query_path(self, path):
    api_path = self.api.path()
    for part in path:
        api_path = api_path.join(part)
    try:
        return list(api_path)
    except LibRouterosError as e:
        self.module.warn('Error while querying path {path}: {error}'.format(path=' '.join(path), error=to_native(e)))
        return []