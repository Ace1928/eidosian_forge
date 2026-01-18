from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.util import files
import six
def remove_metadata(self, metadata_field):
    if metadata_field not in self._track_resource_data:
        raise MetadataNotFoundError(self._resource_name, metadata_field)
    else:
        del self._track_resource_data[metadata_field]