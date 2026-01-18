from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import hashlib
import json
import urllib.parse
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
from googlecloudsdk.core import requests as core_requests
from googlecloudsdk.core import transport
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import transports
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import requests
def get_signing_information_from_file(path, password=None):
    """Loads signing information from a JSON or P12 private key file.

  Args:
    path (str): The location of the file.
    password (str|None): The password used to decrypt encrypted private keys.

  Returns:
    A tuple (client_id: str, key: crypto.PKey), which can be used to sign URLs.
  """
    if password:
        password_bytes = password.encode('utf-8')
    else:
        password_bytes = None
    with files.BinaryFileReader(path) as file:
        raw_data = file.read()
    return get_signing_information_from_json(raw_data, password_bytes)