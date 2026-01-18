from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.resource_map import base
from googlecloudsdk.command_lib.util.resource_map.resource_map import ResourceMap
def get_apitools_apis(self):
    """Returns all apitools collections and associated versions."""
    apitools_apis = {}
    for api in registry.GetAllAPIs():
        if api.name not in apitools_apis:
            apitools_apis[api.name] = []
        apitools_apis[api.name].append(api.version)
    return apitools_apis