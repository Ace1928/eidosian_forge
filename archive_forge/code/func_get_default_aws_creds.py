from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import boto3
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import files
from six.moves import configparser
def get_default_aws_creds():
    """Returns creds from common AWS config file paths.

  Currently does not return "role_arn" because there is no way to extract
  this data from a boto3 Session object.

  Returns:
    Tuple of (access_key_id, secret_access_key, role_arn).
    Each tuple entry can be a string or None.
  """
    credentials = boto3.session.Session().get_credentials()
    if credentials:
        return (credentials.access_key, credentials.secret_key)
    return (None, None)