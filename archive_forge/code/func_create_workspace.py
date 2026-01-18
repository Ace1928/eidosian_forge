from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.dict_transformations import _snake_to_camel, _camel_to_snake
def create_workspace(self, workspace):
    try:
        poller = self.log_analytics_client.workspaces.begin_create_or_update(self.resource_group, self.name, workspace)
        return self.get_poller_result(poller)
    except Exception as exc:
        self.fail('Error when creating workspace {0} - {1}'.format(self.name, exc.message or str(exc)))