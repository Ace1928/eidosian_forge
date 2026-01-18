from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.py import encoding
from apitools.base.py import encoding_helper
from googlecloudsdk.api_lib.storage import metadata_util
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.api_lib.storage.gcs_json import metadata_field_converters
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import gzip_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.resources import gcs_resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import properties
def get_cleared_bucket_fields(request_config):
    """Gets a list of fields to be included in requests despite null values."""
    cleared_fields = []
    resource_args = getattr(request_config, 'resource_args', None)
    if not resource_args:
        return cleared_fields
    if resource_args.cors_file_path == user_request_args_factory.CLEAR or (resource_args.cors_file_path and (not metadata_util.cached_read_yaml_json_file(resource_args.cors_file_path))):
        cleared_fields.append('cors')
    if resource_args.default_encryption_key == user_request_args_factory.CLEAR:
        cleared_fields.append('encryption')
    if resource_args.default_storage_class == user_request_args_factory.CLEAR:
        cleared_fields.append('storageClass')
    if resource_args.labels_file_path == user_request_args_factory.CLEAR:
        cleared_fields.append('labels')
    if resource_args.lifecycle_file_path == user_request_args_factory.CLEAR or (resource_args.lifecycle_file_path and (not metadata_util.cached_read_yaml_json_file(resource_args.lifecycle_file_path))):
        cleared_fields.append('lifecycle')
    if resource_args.log_bucket == user_request_args_factory.CLEAR:
        cleared_fields.append('logging')
    elif resource_args.log_object_prefix == user_request_args_factory.CLEAR:
        cleared_fields.append('logging.logObjectPrefix')
    if resource_args.public_access_prevention == user_request_args_factory.CLEAR:
        cleared_fields.append('iamConfiguration.publicAccessPrevention')
    if resource_args.retention_period == user_request_args_factory.CLEAR:
        cleared_fields.append('retentionPolicy')
    if resource_args.web_error_page == resource_args.web_main_page_suffix == user_request_args_factory.CLEAR:
        cleared_fields.append('website')
    elif resource_args.web_error_page == user_request_args_factory.CLEAR:
        cleared_fields.append('website.notFoundPage')
    elif resource_args.web_main_page_suffix == user_request_args_factory.CLEAR:
        cleared_fields.append('website.mainPageSuffix')
    return cleared_fields