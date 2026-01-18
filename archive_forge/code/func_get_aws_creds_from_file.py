from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import boto3
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import files
from six.moves import configparser
def get_aws_creds_from_file(file_path):
    """Scans file for AWS credentials keys.

  Key fields prefixed with "aws" take precedence.

  Args:
    file_path (str): Path to creds file.

  Returns:
    Tuple of (access_key_id, secret_access_key).
    Each tuple entry can be a string or None.
  """
    creds_dict = get_values_for_keys_from_file(file_path, ['aws_access_key_id', 'aws_secret_access_key', 'access_key_id', 'secret_access_key', 'role_arn'])
    access_key_id = creds_dict.get('aws_access_key_id', creds_dict.get('access_key_id', None))
    secret_access_key = creds_dict.get('aws_secret_access_key', creds_dict.get('secret_access_key', None))
    role_arn = creds_dict.get('role_arn', None)
    return (access_key_id, secret_access_key, role_arn)