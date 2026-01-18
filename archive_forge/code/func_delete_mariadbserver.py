from __future__ import absolute_import, division, print_function
import time
def delete_mariadbserver(self):
    """
        Deletes specified MariaDB Server instance in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the MariaDB Server instance {0}'.format(self.name))
    try:
        response = self.mariadb_client.servers.begin_delete(resource_group_name=self.resource_group, server_name=self.name)
    except Exception as e:
        self.log('Error attempting to delete the MariaDB Server instance.')
        self.fail('Error deleting the MariaDB Server instance: {0}'.format(str(e)))
    return True