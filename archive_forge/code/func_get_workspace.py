from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.dict_transformations import _camel_to_snake
def get_workspace(self):
    try:
        return self.log_analytics_client.workspaces.get(self.resource_group, self.name)
    except ResourceNotFoundError:
        pass
    return None