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
def remove_api(self, api_name):
    """Removes an API from the resource map."""
    if api_name not in self._resource_map_data:
        raise ApiNotFoundError(api_name)
    del self._resource_map_data[api_name]