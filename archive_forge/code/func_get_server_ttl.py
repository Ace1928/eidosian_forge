import os
from ansible.plugins.lookup import LookupBase
import ansible_collections.cloud.common.plugins.module_utils.turbo.common
from ansible_collections.cloud.common.plugins.module_utils.turbo.exceptions import (
def get_server_ttl(variables):
    ttl = os.environ.get('ANSIBLE_TURBO_LOOKUP_TTL', None)
    if ttl is not None:
        return ttl
    for env_var in variables.get('environment', []):
        value = env_var.get('ANSIBLE_TURBO_LOOKUP_TTL', None)
        test_var_int = [isinstance(value, str) and value.isnumeric(), isinstance(value, int)]
        if value is not None and any(test_var_int):
            ttl = value
    return ttl