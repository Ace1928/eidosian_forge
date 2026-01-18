from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import sys
from apitools.base.protorpclite import protojson
from apitools.base.py import encoding
from googlecloudsdk.api_lib.storage import gcs_iam_util
from googlecloudsdk.api_lib.storage import metadata_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import iso_duration
def process_acl_file(file_path, is_bucket=False):
    """Converts ACL file to Apitools objects."""
    acl_dict_list = metadata_util.cached_read_yaml_json_file(file_path)
    acl_class = get_bucket_or_object_acl_class(is_bucket)
    acl_messages = []
    for acl_dict in acl_dict_list:
        acl_messages.append(encoding.DictToMessage(acl_dict, acl_class))
    return acl_messages