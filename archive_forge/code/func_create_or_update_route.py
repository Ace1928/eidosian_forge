from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def create_or_update_route(self, param):
    try:
        poller = self.network_client.routes.begin_create_or_update(self.resource_group, self.route_table_name, self.name, param)
        return self.get_poller_result(poller)
    except Exception as exc:
        self.fail('Error creating or updating route {0} - {1}'.format(self.name, str(exc)))