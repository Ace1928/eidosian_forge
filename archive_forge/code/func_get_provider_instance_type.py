import copy
from enum import Enum
from typing import Any, Dict, List
from ray.autoscaler._private.util import hash_runtime_conf, prepare_config
def get_provider_instance_type(self, instance_type_name: str) -> str:
    provider = self.provider
    node_config = self.get_node_type_specific_config(instance_type_name, 'node_config')
    if provider in [Provider.AWS, Provider.ALIYUN]:
        return node_config.get('InstanceType', '')
    elif provider == Provider.AZURE:
        return node_config.get('azure_arm_parameters', {}).get('vmSize', '')
    elif provider == Provider.GCP:
        return node_config.get('machineType', '')
    elif provider in [Provider.KUBERAY, Provider.LOCAL, Provider.UNKNOWN]:
        return ''
    else:
        raise ValueError(f'Unknown provider {provider}')