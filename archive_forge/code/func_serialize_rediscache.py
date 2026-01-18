from __future__ import absolute_import, division, print_function
import re
def serialize_rediscache(self, rediscache):
    """
        Convert an Azure Cache for Redis object to dict.
        :param rediscache: Azure Cache for Redis object
        :return: dict
        """
    new_result = dict(id=rediscache.id, resource_group=re.sub('\\/.*', '', re.sub('.*resourceGroups\\/', '', rediscache.id)), name=rediscache.name, location=rediscache.location, provisioning_state=rediscache.provisioning_state, configuration=rediscache.redis_configuration, tenant_settings=rediscache.tenant_settings, minimum_tls_version=rediscache.minimum_tls_version, public_network_access=rediscache.public_network_access, redis_version=rediscache.redis_version, shard_count=rediscache.shard_count, enable_non_ssl_port=rediscache.enable_non_ssl_port, static_ip=rediscache.static_ip, subnet=rediscache.subnet_id, host_name=rediscache.host_name, tags=rediscache.tags)
    if rediscache.sku:
        new_result['sku'] = dict(name=rediscache.sku.name.lower(), size=rediscache.sku.family + str(rediscache.sku.capacity))
    if self.return_access_keys:
        access_keys = self.list_keys()
        if access_keys:
            new_result['access_keys'] = dict(primary=access_keys.primary_key, secondary=access_keys.secondary_key)
    return new_result