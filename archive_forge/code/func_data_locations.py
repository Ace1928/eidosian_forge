from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from apitools.base.py import encoding
from googlecloudsdk.command_lib.storage.resources import full_resource_formatter
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
@property
def data_locations(self):
    if self.custom_placement_config:
        return self.custom_placement_config.get('dataLocations')
    return None