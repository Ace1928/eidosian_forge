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
def check_if_will_use_fast_crc32c(install_if_missing=False):
    return crc32c.IS_FAST_GOOGLE_CRC32C_AVAILABLE or (properties.VALUES.storage.use_gcloud_crc32c.GetBool() is not False and _check_if_gcloud_crc32c_available(install_if_missing))