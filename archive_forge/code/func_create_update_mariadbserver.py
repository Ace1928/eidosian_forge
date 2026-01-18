from __future__ import absolute_import, division, print_function
import time
def create_update_mariadbserver(self):
    """
        Creates or updates MariaDB Server with the specified configuration.

        :return: deserialized MariaDB Server instance state dictionary
        """
    self.log('Creating / Updating the MariaDB Server instance {0}'.format(self.name))
    try:
        self.parameters['tags'] = self.tags
        if self.to_do == Actions.Create:
            response = self.mariadb_client.servers.begin_create(resource_group_name=self.resource_group, server_name=self.name, parameters=self.parameters)
        else:
            self.parameters.update(self.parameters.pop('properties', {}))
            response = self.mariadb_client.servers.begin_update(resource_group_name=self.resource_group, server_name=self.name, parameters=self.parameters)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error attempting to create the MariaDB Server instance.')
        self.fail('Error creating the MariaDB Server instance: {0}'.format(str(exc)))
    return response.as_dict()