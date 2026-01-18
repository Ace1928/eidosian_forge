from __future__ import absolute_import, division, print_function
import time
def create_update_postgresqldatabase(self):
    """
        Creates or updates PostgreSQL Database with the specified configuration.

        :return: deserialized PostgreSQL Database instance state dictionary
        """
    self.log('Creating / Updating the PostgreSQL Database instance {0}'.format(self.name))
    try:
        response = self.postgresql_client.databases.begin_create_or_update(resource_group_name=self.resource_group, server_name=self.server_name, database_name=self.name, parameters=self.parameters)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error attempting to create the PostgreSQL Database instance.')
        self.fail('Error creating the PostgreSQL Database instance: {0}'.format(str(exc)))
    return response.as_dict()