from __future__ import absolute_import, division, print_function
import time
def regenerate_rediscache_key(self):
    """
        Regenerate key of redis cache instance in the specified subscription and resource group.

        :return: True
        """
    self.log('Regenerate key of redis cache instance {0}'.format(self.name))
    try:
        params = RedisRegenerateKeyParameters(key_type=self.regenerate_key['key_type'].title())
        response = self._client.redis.regenerate_key(resource_group_name=self.resource_group, name=self.name, parameters=params)
        return response.as_dict()
    except Exception as e:
        self.log('Error attempting to regenerate key of redis cache instance.')
        self.fail('Error regenerate key of redis cache instance: {0}'.format(str(e)))