from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
import math
import os
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import scaled_integer
def get_temporary_component_resource(source_resource, destination_resource, random_prefix, component_id):
    """Gets a temporary component destination resource for a composite upload.

  Args:
    source_resource (resource_reference.FileObjectResource): The upload source.
    destination_resource (resource_reference.ObjectResource|UnknownResource):
      The upload destination.
    random_prefix (str): Added to temporary component names to avoid collisions
      between different instances of the CLI uploading to the same destination.
    component_id (int): An id that's not shared by any other component in this
      transfer.

  Returns:
    A resource_reference.UnknownResource representing the component's
    destination.
  """
    component_object_name = _get_temporary_component_name(source_resource, destination_resource, random_prefix, component_id)
    destination_url = destination_resource.storage_url
    component_url = storage_url.CloudUrl(destination_url.scheme, destination_url.bucket_name, component_object_name)
    return resource_reference.UnknownResource(component_url)