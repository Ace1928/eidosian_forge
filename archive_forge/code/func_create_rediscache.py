from __future__ import absolute_import, division, print_function
import time
def create_rediscache(self):
    """
        Creates Azure Cache for Redis instance with the specified configuration.

        :return: deserialized Azure Cache for Redis instance state dictionary
        """
    self.log('Creating Azure Cache for Redis instance {0}'.format(self.name))
    try:
        redis_config = dict()
        for key in self.redis_configuration_properties:
            if getattr(self, key, None):
                redis_config[underline_to_hyphen(key)] = underline_to_hyphen(getattr(self, key))
        params = RedisCreateParameters(location=self.location, sku=Sku(name=self.sku['name'].title(), family=self.sku['size'][0], capacity=self.sku['size'][1:]), tags=self.tags, redis_configuration=redis_config, enable_non_ssl_port=self.enable_non_ssl_port, tenant_settings=self.tenant_settings, minimum_tls_version=self.minimum_tls_version, public_network_access=self.public_network_access, redis_version=self.redis_version, shard_count=self.shard_count, subnet_id=self.subnet, static_ip=self.static_ip)
        response = self._client.redis.begin_create(resource_group_name=self.resource_group, name=self.name, parameters=params)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
        if self.wait_for_provisioning:
            self.wait_for_redis_running()
    except Exception as exc:
        self.log('Error attempting to create the Azure Cache for Redis instance.')
        self.fail('Error creating the Azure Cache for Redis instance: {0}'.format(str(exc)))
    return rediscache_to_dict(response)