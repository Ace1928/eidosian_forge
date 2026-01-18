from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import copy
import re
from googlecloudsdk.api_lib.storage import metadata_util
from googlecloudsdk.api_lib.storage import xml_metadata_field_converters
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.resources import gcs_resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import s3_resource_reference
from googlecloudsdk.core import log
def get_object_resource_from_xml_response(scheme, object_dict, bucket_name, object_name=None, acl_dict=None):
    """Creates resource_reference.S3ObjectResource from S3 API response.

  Args:
    scheme (storage_url.ProviderPrefix): Prefix used for provider URLs.
    object_dict (dict): Dictionary representing S3 API response.
    bucket_name (str): Bucket response is relevant to.
    object_name (str|None): Object if relevant to query.
    acl_dict (dict|None): Response from S3 get_object_acl API call.

  Returns:
    resource_reference.S3ObjectResource populated with data.
  """
    object_url = _get_object_url_from_xml_response(scheme, object_dict, bucket_name, object_name or object_dict['Key'])
    if 'Size' in object_dict:
        size = object_dict.get('Size')
    else:
        size = object_dict.get('ContentLength')
    encryption_algorithm = object_dict.get('ServerSideEncryption', object_dict.get('SSECustomerAlgorithm'))
    etag = _get_etag(object_dict)
    if acl_dict:
        raw_acl_data = acl_dict
    else:
        raw_acl_data = object_dict.get('ACL')
    if raw_acl_data:
        object_dict['ACL'] = raw_acl_data
    acl = _get_error_or_value(raw_acl_data)
    object_resource = _SCHEME_TO_OBJECT_RESOURCE_DICT[scheme](object_url, acl=acl, cache_control=object_dict.get('CacheControl'), component_count=object_dict.get('PartsCount'), content_disposition=object_dict.get('ContentDisposition'), content_encoding=object_dict.get('ContentEncoding'), content_language=object_dict.get('ContentLanguage'), content_type=object_dict.get('ContentType'), creation_time=object_dict.get('LastModified'), custom_fields=object_dict.get('Metadata'), encryption_algorithm=encryption_algorithm, etag=etag, kms_key=object_dict.get('SSEKMSKeyId'), md5_hash=_get_md5_hash_from_etag(etag, object_url), metadata=object_dict, size=size, storage_class=object_dict.get('StorageClass'), update_time=object_dict.get('LastModified'))
    if scheme == storage_url.ProviderPrefix.GCS:
        object_resource.crc32c_hash = _get_crc32c_hash_from_object_dict(object_dict)
    return object_resource