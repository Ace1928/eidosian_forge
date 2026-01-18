from __future__ import absolute_import, division, print_function
import time
def get_rediscache(self):
    """
        Gets the properties of the specified Azure Cache for Redis instance.

        :return: Azure Cache for Redis instance state dictionary
        """
    self.log('Checking if the Azure Cache for Redis instance {0} is present'.format(self.name))
    response = None
    try:
        response = self._client.redis.get(resource_group_name=self.resource_group, name=self.name)
        self.log('Response : {0}'.format(response))
        self.log('Azure Cache for Redis instance : {0} found'.format(response.name))
        return rediscache_to_dict(response)
    except ResourceNotFoundError:
        self.log("Didn't find Azure Cache for Redis {0} in resource group {1}".format(self.name, self.resource_group))
    return False