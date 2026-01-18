from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage.resources import full_resource_formatter as base
from googlecloudsdk.command_lib.storage.resources import shim_format_util
def format_bucket(self, bucket_resource):
    """See super class."""
    shim_format_util.replace_autoclass_value_with_prefixed_time(bucket_resource)
    shim_format_util.replace_bucket_values_with_present_string(bucket_resource)
    return base.get_formatted_string(bucket_resource, _BUCKET_DISPLAY_TITLES_AND_DEFAULTS)