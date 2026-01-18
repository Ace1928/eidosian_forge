from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import enum
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import fast_crc32c_util
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import hashing
def get_bytes_from_base64_string(hash_string):
    """Takes base64-encoded string and returns bytes."""
    hash_bytes = hash_string.encode('utf-8')
    return base64.b64decode(hash_bytes)