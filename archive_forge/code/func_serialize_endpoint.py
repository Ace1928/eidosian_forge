from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _camel_to_snake
import re
def serialize_endpoint(endpoint):
    result = dict(id=endpoint.id, name=endpoint.name, target_resource_id=endpoint.target_resource_id, target=endpoint.target, status=endpoint.endpoint_status, weight=endpoint.weight, priority=endpoint.priority, location=endpoint.endpoint_location, min_child_endpoints=endpoint.min_child_endpoints, geo_mapping=endpoint.geo_mapping)
    if endpoint.type:
        result['type'] = _camel_to_snake(endpoint.type.split('/')[-1])
    return result