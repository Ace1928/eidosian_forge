from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import errno
import json
import uuid
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from apitools.base.py import transfer as apitools_transfer
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import errors as cloud_errors
from googlecloudsdk.api_lib.storage import gcs_iam_util
from googlecloudsdk.api_lib.storage import headers_util
from googlecloudsdk.api_lib.storage.gcs_json import download
from googlecloudsdk.api_lib.storage.gcs_json import error_util
from googlecloudsdk.api_lib.storage.gcs_json import metadata_util
from googlecloudsdk.api_lib.storage.gcs_json import patch_apitools_messages
from googlecloudsdk.api_lib.storage.gcs_json import upload
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors as command_errors
from googlecloudsdk.command_lib.storage import gzip_util
from googlecloudsdk.command_lib.storage import posix_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import tracker_file_util
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.resources import gcs_resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.command_lib.storage.tasks.cp import download_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core.credentials import transports
from googlecloudsdk.core.util import scaled_integer
import six
from six.moves import urllib
@error_util.catch_http_error_raise_gcs_api_error()
def patch_bucket(self, bucket_resource, request_config, fields_scope=cloud_api.FieldsScope.NO_ACL):
    """See super class."""
    projection = self._get_projection(fields_scope, self.messages.StorageBucketsPatchRequest)
    metadata = getattr(bucket_resource, 'metadata', None) or metadata_util.get_apitools_metadata_from_url(bucket_resource.storage_url)
    metadata_util.update_bucket_metadata_from_request_config(metadata, request_config)
    cleared_fields = metadata_util.get_cleared_bucket_fields(request_config)
    if metadata.defaultObjectAcl and metadata.defaultObjectAcl[0] == metadata_util.PRIVATE_DEFAULT_OBJECT_ACL:
        cleared_fields.append('defaultObjectAcl')
        metadata.defaultObjectAcl = []
    if request_config.predefined_acl_string:
        cleared_fields.append('acl')
        predefined_acl = getattr(self.messages.StorageBucketsPatchRequest.PredefinedAclValueValuesEnum, request_config.predefined_acl_string)
    else:
        predefined_acl = None
    if request_config.predefined_default_object_acl_string:
        cleared_fields.append('defaultObjectAcl')
        predefined_default_object_acl = getattr(self.messages.StorageBucketsPatchRequest.PredefinedDefaultObjectAclValueValuesEnum, request_config.predefined_default_object_acl_string)
    else:
        predefined_default_object_acl = None
    apitools_request = self.messages.StorageBucketsPatchRequest(bucket=bucket_resource.storage_url.bucket_name, bucketResource=metadata, projection=projection, ifMetagenerationMatch=request_config.precondition_metageneration_match, predefinedAcl=predefined_acl, predefinedDefaultObjectAcl=predefined_default_object_acl)
    with self.client.IncludeFields(cleared_fields):
        return metadata_util.get_bucket_resource_from_metadata(self.client.buckets.Patch(apitools_request))