from __future__ import absolute_import, division, print_function
import time
def create_update_mysqlserver(self):
    """
        Creates or updates MySQL Server with the specified configuration.

        :return: deserialized MySQL Server instance state dictionary
        """
    self.log('Creating / Updating the MySQL Server instance {0}'.format(self.name))
    try:
        self.parameters['tags'] = self.tags
        if self.to_do == Actions.Create:
            response = self.mysql_client.servers.begin_create(resource_group_name=self.resource_group, server_name=self.name, parameters=self.parameters)
        else:
            self.parameters.update(self.parameters.pop('properties', {}))
            response = self.mysql_client.servers.begin_update(resource_group_name=self.resource_group, server_name=self.name, parameters=self.parameters)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error attempting to create the MySQL Server instance.')
        self.fail('Error creating the MySQL Server instance: {0}'.format(str(exc)))
    return response.as_dict()