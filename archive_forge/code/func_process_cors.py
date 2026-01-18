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
def process_cors(file_path):
    """Converts CORS file to Apitools objects."""
    if file_path == user_request_args_factory.CLEAR:
        return []
    cors_dict_list = metadata_util.cached_read_yaml_json_file(file_path)
    if not cors_dict_list:
        return []
    cors_messages = []
    messages = apis.GetMessagesModule('storage', 'v1')
    for cors_dict in cors_dict_list:
        cors_messages.append(encoding.DictToMessage(cors_dict, messages.Bucket.CorsValueListEntry))
    return cors_messages