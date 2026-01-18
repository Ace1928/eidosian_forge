from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.command_lib.util.resource_map.declarative import declarative_map
from googlecloudsdk.core import exceptions
def populate_name_mappings(self, resource_map):
    """Populates name maps for constant time access to resources."""
    self.ai_map = {}
    self.krm_map = {}
    self.collection_map = {}
    for api in resource_map:
        for resource in api:
            wrapped_resource = self.ResourceNameTranslatorWrapper(resource)
            if resource.has_metadata_field('asset_inventory_type'):
                self.ai_map[resource.asset_inventory_type] = wrapped_resource
                self.krm_map[KrmKind(resource.krm_group, resource.krm_kind)] = wrapped_resource
                self.collection_map[resource.get_full_collection_name()] = wrapped_resource