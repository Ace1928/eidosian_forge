from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.core.util import scaled_integer
def replace_bucket_values_with_present_string(bucket_resource):
    """Updates fields with complex data to a simple 'Present' string."""
    for field in _BUCKET_FIELDS_WITH_PRESENT_VALUE:
        value = getattr(bucket_resource, field)
        if value and (not isinstance(value, errors.CloudApiError)):
            setattr(bucket_resource, field, PRESENT_STRING)