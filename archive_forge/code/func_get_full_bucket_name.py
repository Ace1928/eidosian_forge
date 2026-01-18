from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import errors as cloud_errors
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.core import log
def get_full_bucket_name(bucket_name):
    """Returns the bucket resource name as expected by gRPC API."""
    return 'projects/_/buckets/{}'.format(bucket_name)