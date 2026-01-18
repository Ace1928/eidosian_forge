from __future__ import absolute_import, division, print_function
import time
def get_mysqldatabase(self):
    """
        Gets the properties of the specified MySQL Database.

        :return: deserialized MySQL Database instance state dictionary
        """
    self.log('Checking if the MySQL Database instance {0} is present'.format(self.name))
    found = False
    try:
        response = self.mysql_client.databases.get(resource_group_name=self.resource_group, server_name=self.server_name, database_name=self.name)
        found = True
        self.log('Response : {0}'.format(response))
        self.log('MySQL Database instance : {0} found'.format(response.name))
    except ResourceNotFoundError as e:
        self.log('Did not find the MySQL Database instance.')
    if found is True:
        return response.as_dict()
    return False