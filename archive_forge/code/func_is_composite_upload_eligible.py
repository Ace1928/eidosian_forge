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
def is_composite_upload_eligible(source_resource, destination_resource, user_request_args=None):
    """Checks if parallel composite upload should be performed.

  Logs tailored warning based on user configuration and the context
  of the operation.
  Informs user about configuration options they may want to set.
  In order to avoid repeated warning raised for each task,
  this function updates the storage/parallel_composite_upload_enabled
  so that the warnings are logged only once.

  Args:
    source_resource (FileObjectResource): The source file
      resource to be uploaded.
    destination_resource(CloudResource|UnknownResource):
      Destination resource to which the files should be uploaded.
    user_request_args (UserRequestArgs|None): Values for RequestConfig.

  Returns:
    True if the parallel composite upload can be performed. However, this does
    not guarantee that parallel composite upload will be performed as the
    parallelism check can happen only after the task executor starts running
    because it sets the process_count and thread_count. We also let the task
    determine the component count.
  """
    composite_upload_enabled = properties.VALUES.storage.parallel_composite_upload_enabled.GetBool()
    if composite_upload_enabled is False:
        return False
    if not isinstance(source_resource, resource_reference.FileObjectResource):
        return False
    try:
        if source_resource.size is None or source_resource.size < scaled_integer.ParseInteger(properties.VALUES.storage.parallel_composite_upload_threshold.Get()):
            return False
    except OSError as e:
        log.warning('Size cannot be determined for resource: %s. Error: %s', source_resource, e)
        return False
    compatibility_check_required = properties.VALUES.storage.parallel_composite_upload_compatibility_check.GetBool()
    if composite_upload_enabled and (not compatibility_check_required):
        return True
    api_capabilities = api_factory.get_capabilities(destination_resource.storage_url.scheme)
    if cloud_api.Capability.COMPOSE_OBJECTS not in api_capabilities:
        properties.VALUES.storage.parallel_composite_upload_enabled.Set(False)
        return False
    if compatibility_check_required:
        can_perform_composite_upload = is_destination_composite_upload_compatible(destination_resource, user_request_args)
        properties.VALUES.storage.parallel_composite_upload_compatibility_check.Set(False)
    else:
        can_perform_composite_upload = True
    if can_perform_composite_upload and composite_upload_enabled is None:
        log.warning(textwrap.fill('Parallel composite upload was turned ON to get the best performance on uploading large objects. If you would like to opt-out and instead perform a normal upload, run:\n`gcloud config set storage/parallel_composite_upload_enabled False`\nIf you would like to disable this warning, run:\n`gcloud config set storage/parallel_composite_upload_enabled True`\nNote that with parallel composite uploads, your object might be uploaded as a composite object (https://cloud.google.com/storage/docs/composite-objects), which means that any user who downloads your object will need to use crc32c checksums to verify data integrity. gcloud storage is capable of computing crc32c checksums, but this might pose a problem for other clients.') + '\n')
    properties.VALUES.storage.parallel_composite_upload_enabled.Set(can_perform_composite_upload)
    return can_perform_composite_upload