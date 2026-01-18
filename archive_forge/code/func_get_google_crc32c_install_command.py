from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import struct
import textwrap
from googlecloudsdk.command_lib import info_holder
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.util import crc32c
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def get_google_crc32c_install_command():
    """Returns the command to install google-crc32c library.

  This will typically only be called if gcloud-crc32c is missing and can't be
  installed for some reason. It requires user intervention which is why it's
  not a preferred option.
  """
    sdk_info = info_holder.InfoHolder()
    sdk_root = sdk_info.installation.sdk_root
    if sdk_root:
        third_party_path = os.path.join(sdk_root, 'lib', 'third_party')
        return '{} -m pip install google-crc32c --upgrade --target {}'.format(sdk_info.basic.python_location, third_party_path)
    return None