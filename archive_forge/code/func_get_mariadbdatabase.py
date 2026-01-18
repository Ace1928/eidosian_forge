from __future__ import absolute_import, division, print_function
import time
def get_mariadbdatabase(self):
    """
        Gets the properties of the specified MariaDB Database.

        :return: deserialized MariaDB Database instance state dictionary
        """
    self.log('Checking if the MariaDB Database instance {0} is present'.format(self.name))
    found = False
    try:
        response = self.mariadb_client.databases.get(resource_group_name=self.resource_group, server_name=self.server_name, database_name=self.name)
        found = True
        self.log('Response : {0}'.format(response))
        self.log('MariaDB Database instance : {0} found'.format(response.name))
    except ResourceNotFoundError as e:
        self.log('Did not find the MariaDB Database instance.')
    if found is True:
        return response.as_dict()
    return False