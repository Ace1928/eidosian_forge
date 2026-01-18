from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import errors as cloud_errors
from googlecloudsdk.api_lib.storage.gcs_grpc import download
from googlecloudsdk.api_lib.storage.gcs_grpc import metadata_util
from googlecloudsdk.api_lib.storage.gcs_grpc import upload
from googlecloudsdk.api_lib.storage.gcs_json import client as gcs_json_client
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import gzip_util
from googlecloudsdk.command_lib.storage import tracker_file_util
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.command_lib.storage.tasks.cp import download_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import scaled_integer
def copy_object(self, source_resource, destination_resource, request_config, posix_to_set=None, progress_callback=None, should_deep_copy_metadata=False):
    """See super class."""
    self._get_gapic_client()
    destination_metadata = getattr(destination_resource, 'metadata', None)
    if not destination_metadata:
        destination_metadata = metadata_util.get_grpc_metadata_from_url(destination_resource.storage_url, self._gapic_client.types)
    if source_resource.metadata:
        destination_metadata = metadata_util.copy_object_metadata(source_metadata=source_resource.metadata, destination_metadata=destination_metadata, request_config=request_config, should_deep_copy=should_deep_copy_metadata)
    metadata_util.update_object_metadata_from_request_config(destination_metadata, request_config, posix_to_set=posix_to_set)
    if request_config.predefined_acl_string and request_config.predefined_acl_string in self.ALLOWED_PREDFINED_DESTINATION_ACL_VALUES:
        predefined_acl = request_config.predefined_acl_string
    else:
        predefined_acl = None
    if source_resource.generation is None:
        source_generation = None
    else:
        source_generation = int(source_resource.generation)
    tracker_file_path = tracker_file_util.get_tracker_file_path(destination_resource.storage_url, tracker_file_util.TrackerFileType.REWRITE, source_url=source_resource.storage_url)
    rewrite_parameters_hash = tracker_file_util.hash_gcs_rewrite_parameters_for_tracker_file(source_object_resource=source_resource, destination_object_resource=destination_resource, destination_metadata=destination_metadata, request_config=request_config)
    resume_rewrite_token = tracker_file_util.get_rewrite_token_from_tracker_file(tracker_file_path, rewrite_parameters_hash)
    if resume_rewrite_token:
        log.debug('Found rewrite token. Resuming copy.')
    else:
        log.debug('No rewrite token found. Starting copy from scratch.')
    max_bytes_per_call = scaled_integer.ParseInteger(properties.VALUES.storage.copy_chunk_size.Get())
    with self._encryption_headers_for_rewrite_call_context(request_config):
        while True:
            request = self._gapic_client.types.RewriteObjectRequest(source_bucket=source_resource.storage_url.bucket_name, source_object=source_resource.storage_url.object_name, destination_bucket=destination_resource.storage_url.bucket_name, destination_name=destination_resource.storage_url.object_name, destination=destination_metadata, source_generation=source_generation, if_generation_match=copy_util.get_generation_match_value(request_config), if_metageneration_match=request_config.precondition_metageneration_match, destination_predefined_acl=predefined_acl, rewrite_token=resume_rewrite_token, max_bytes_rewritten_per_call=max_bytes_per_call)
            encryption_key = getattr(request_config.resource_args, 'encryption_key', None)
            if encryption_key and encryption_key != user_request_args_factory.CLEAR and (encryption_key.type == encryption_util.KeyType.CMEK):
                request.destination_kms_key = encryption_key.key
            rewrite_response = self._gapic_client.storage.rewrite_object(request)
            processed_bytes = rewrite_response.total_bytes_rewritten
            if progress_callback:
                progress_callback(processed_bytes)
            if rewrite_response.done:
                break
            if not resume_rewrite_token:
                resume_rewrite_token = rewrite_response.rewrite_token
                if source_resource.size >= scaled_integer.ParseInteger(properties.VALUES.storage.resumable_threshold.Get()):
                    tracker_file_util.write_rewrite_tracker_file(tracker_file_path, rewrite_parameters_hash, rewrite_response.rewrite_token)
    tracker_file_util.delete_tracker_file(tracker_file_path)
    return metadata_util.get_object_resource_from_grpc_object(rewrite_response.resource)