import copy
from enum import Enum
from typing import Any, Dict, List
from ray.autoscaler._private.util import hash_runtime_conf, prepare_config
@property
def provider(self) -> Provider:
    provider_str = self._node_configs.get('provider', {}).get('type', '')
    if provider_str == 'local':
        return Provider.LOCAL
    elif provider_str == 'aws':
        return Provider.AWS
    elif provider_str == 'azure':
        return Provider.AZURE
    elif provider_str == 'gcp':
        return Provider.GCP
    elif provider_str == 'aliyun':
        return Provider.ALIYUN
    elif provider_str == 'kuberay':
        return Provider.KUBERAY
    else:
        return Provider.UNKNOWN