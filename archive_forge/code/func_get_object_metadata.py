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
def get_object_metadata(self, bucket_name, object_name, request_config=None, generation=None, fields_scope=cloud_api.FieldsScope.NO_ACL, soft_deleted=False):
    """See super class."""
    if generation:
        generation = int(generation)
    projection = self._get_projection(fields_scope, self.messages.StorageObjectsGetRequest)
    global_params = None
    if fields_scope == cloud_api.FieldsScope.SHORT:
        global_params = self.messages.StandardQueryParameters(fields='bucket,name,size,generation')
    decryption_key = getattr(getattr(request_config, 'resource_args', None), 'decryption_key', None)
    with self._encryption_headers_context(decryption_key):
        object_metadata = self.client.objects.Get(self.messages.StorageObjectsGetRequest(bucket=bucket_name, object=object_name, generation=generation, projection=projection, softDeleted=True if soft_deleted else None), global_params=global_params)
    return metadata_util.get_object_resource_from_metadata(object_metadata)