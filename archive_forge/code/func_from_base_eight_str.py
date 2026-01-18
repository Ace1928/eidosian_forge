from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import os
import stat
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import log
from googlecloudsdk.core.cache import function_result_cache
from googlecloudsdk.core.util import platforms
@classmethod
def from_base_eight_str(cls, base_eight_str):
    """Initializes class from base eight (octal) string. E.g. '111'."""
    return PosixMode(convert_base_eight_str_to_base_ten_int(base_eight_str), base_eight_str)