from __future__ import absolute_import, division, print_function
import time
def delete_postgresqldatabase(self):
    """
        Deletes specified PostgreSQL Database instance in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the PostgreSQL Database instance {0}'.format(self.name))
    try:
        response = self.postgresql_client.databases.begin_delete(resource_group_name=self.resource_group, server_name=self.server_name, database_name=self.name)
    except Exception as e:
        self.log('Error attempting to delete the PostgreSQL Database instance.')
        self.fail('Error deleting the PostgreSQL Database instance: {0}'.format(str(e)))
    return True