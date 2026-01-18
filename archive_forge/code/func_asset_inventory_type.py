from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.command_lib.util.resource_map.declarative import declarative_map
from googlecloudsdk.core import exceptions
@property
def asset_inventory_type(self):
    return self._resource.asset_inventory_type