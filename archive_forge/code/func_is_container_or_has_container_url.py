from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage.resources import resource_util
def is_container_or_has_container_url(resource):
    """Returns if resource is a known or unverified container resource."""
    if isinstance(resource, UnknownResource):
        return resource.storage_url.is_bucket()
    return resource.is_container()