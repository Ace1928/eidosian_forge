from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.core.util import scaled_integer
def replace_time_values_with_gsutil_style_strings(resource):
    """Updates fields in gcloud time format to gsutil time format."""
    for key in ('creation_time', 'custom_time', 'noncurrent_time', 'retention_expiration', 'storage_class_update_time', 'update_time'):
        gcloud_datetime = getattr(resource, key, None)
        if gcloud_datetime is not None:
            setattr(resource, key, _gsutil_format_datetime_string(gcloud_datetime))