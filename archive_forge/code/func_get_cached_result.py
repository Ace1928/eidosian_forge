from ansible.plugins.inventory import BaseInventoryPlugin
from ansible.plugins.inventory import Cacheable
from ansible.plugins.inventory import Constructable
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.plugin_utils.base import AWSPluginBase
from ansible_collections.amazon.aws.plugins.plugin_utils.botocore import AnsibleBotocoreError
def get_cached_result(self, path, cache):
    if not cache:
        return (False, None)
    if not self.get_option('cache'):
        return (False, None)
    cache_key = self.get_cache_key(path)
    try:
        cached_value = self._cache[cache_key]
    except KeyError:
        return (False, None)
    return (True, cached_value)