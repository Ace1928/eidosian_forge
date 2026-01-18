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
def get_posix_attributes_from_file(file_path, preserve_symlinks=False):
    """Takes file path and returns PosixAttributes object."""
    follow_symlinks = not preserve_symlinks or os.stat not in os.supports_follow_symlinks
    mode, _, _, _, uid, gid, _, atime, mtime, _ = os.stat(file_path, follow_symlinks=follow_symlinks)
    return PosixAttributes(atime, mtime, uid, gid, PosixMode.from_base_ten_int(mode))