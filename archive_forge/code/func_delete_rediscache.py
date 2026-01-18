from __future__ import absolute_import, division, print_function
import time
def delete_rediscache(self):
    """
        Deletes specified Azure Cache for Redis instance in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the Azure Cache for Redis instance {0}'.format(self.name))
    try:
        self._client.redis.begin_delete(resource_group_name=self.resource_group, name=self.name)
    except Exception as e:
        self.log('Error attempting to delete the Azure Cache for Redis instance.')
        self.fail('Error deleting the Azure Cache for Redis instance: {0}'.format(str(e)))
    return True