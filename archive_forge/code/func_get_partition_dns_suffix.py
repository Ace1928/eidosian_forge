import copy
import logging
import re
from enum import Enum
from botocore import UNSIGNED, xform_name
from botocore.auth import AUTH_TYPE_MAPS, HAS_CRT
from botocore.crt import CRT_SUPPORTED_AUTH_TYPES
from botocore.endpoint_provider import EndpointProvider
from botocore.exceptions import (
from botocore.utils import ensure_boolean, instance_cache
def get_partition_dns_suffix(self, partition_name, endpoint_variant_tags=None):
    for partition in self._endpoint_data['partitions']:
        if partition['partition'] == partition_name:
            if endpoint_variant_tags:
                variant = self._retrieve_variant_data(partition.get('defaults'), endpoint_variant_tags)
                if variant and 'dnsSuffix' in variant:
                    return variant['dnsSuffix']
            else:
                return partition['dnsSuffix']
    return None