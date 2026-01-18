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
@function_result_cache.lru(maxsize=1)
def get_system_posix_data():
    """Gets POSIX info that should only be fetched once."""
    if platforms.OperatingSystem.IsWindows():
        return SystemPosixData(None, None)
    default_mode = _get_default_mode()
    user_groups = _get_user_groups()
    return SystemPosixData(default_mode, user_groups)