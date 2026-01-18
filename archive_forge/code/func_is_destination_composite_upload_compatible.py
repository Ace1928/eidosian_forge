from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import scaled_integer
def is_destination_composite_upload_compatible(destination_resource, user_request_args):
    """Checks if destination bucket is compatible for parallel composite upload.

  This function performs a GET bucket call to determine if the bucket's default
  storage class and retention period meet the criteria.

  Args:
    destination_resource(CloudResource|UnknownResource):
      Destination resource to which the files should be uploaded.
    user_request_args (UserRequestArgs|None): Values from user flags.

  Returns:
    True if the bucket satisfies the storage class and retention policy
    criteria.

  """
    api_client = api_factory.get_api(destination_resource.storage_url.scheme)
    try:
        bucket_resource = api_client.get_bucket(destination_resource.storage_url.bucket_name)
    except errors.CloudApiError as e:
        status = getattr(e, 'status_code', None)
        if status in (401, 403):
            log.error('Cannot check if the destination bucket is compatible for running parallel composite uploads as the user does not permission to perform GET operation on the bucket. The operation will be performed without parallel composite upload feature and hence might perform relatively slower.')
            return False
        else:
            raise
    resource_args = getattr(user_request_args, 'resource_args', None)
    object_storage_class = getattr(resource_args, 'storage_class', None)
    if bucket_resource.retention_period is not None:
        reason = 'Destination bucket has retention period set'
    elif bucket_resource.default_event_based_hold:
        reason = 'Destination bucket has event-based hold set'
    elif getattr(resource_args, 'event_based_hold', None):
        reason = 'Object will be created with event-based hold'
    elif getattr(resource_args, 'temporary_hold', None):
        reason = 'Object will be created with temporary hold'
    elif bucket_resource.default_storage_class != _STANDARD_STORAGE_CLASS and object_storage_class != _STANDARD_STORAGE_CLASS:
        reason = 'Destination has a default storage class other than "STANDARD"'
    elif object_storage_class not in (None, _STANDARD_STORAGE_CLASS):
        reason = 'Object will be created with a storage class other than "STANDARD"'
    else:
        return True
    log.warning('{}, hence parallel composite upload will not be performed. If you would like to disable this check, run: gcloud config set storage/parallel_composite_upload_compatibility_check=False'.format(reason))
    return False