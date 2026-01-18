from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.declarative.clients import kcc_client
from googlecloudsdk.command_lib.util.resource_map import base
from googlecloudsdk.command_lib.util.resource_map import resource_map_update_util
from googlecloudsdk.command_lib.util.resource_map.declarative import declarative_map
from googlecloudsdk.core import name_parsing
def get_apitools_collections():
    """Returns all apitools collections and associated versions."""
    collection_api_names = set()
    collection_api_versions = {}
    for api in registry.GetAllAPIs():
        collection_api_names.add(api.name)
        if api.name not in collection_api_versions:
            collection_api_versions[api.name] = []
        collection_api_versions[api.name].append(api.version)
    return (collection_api_names, collection_api_versions)