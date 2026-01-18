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
def process_iam_file(file_path, custom_etag=None):
    """Converts IAM file to Apitools objects."""
    if file_path == '-' and properties.VALUES.storage.run_by_gsutil_shim.GetBool():
        policy_dict = metadata_util.read_yaml_json_from_string(sys.stdin.read())
    else:
        policy_dict = metadata_util.cached_read_yaml_json_file(file_path)
    policy_dict['version'] = gcs_iam_util.IAM_POLICY_VERSION
    if custom_etag is not None:
        policy_dict['etag'] = custom_etag
    policy_string = json.dumps(policy_dict)
    messages = apis.GetMessagesModule('storage', 'v1')
    policy_object = protojson.decode_message(messages.Policy, policy_string)
    return policy_object