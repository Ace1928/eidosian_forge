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
def set_posix_attributes_on_file_if_valid(system_posix_data, source_resource, destination_resource, known_source_posix=None, known_destination_posix=None, preserve_symlinks=False):
    """Sets custom POSIX attributes on file if the final metadata will be valid.

  This function is typically called after downloads.
  `raise_if_invalid_file_permissions` should have been called before initiating
  a download, but we call it again here to be safe.

  Args:
    system_posix_data (SystemPosixData): System-wide POSIX. Helps fill in
      missing data and determine validity of result.
    source_resource (resource_reference.ObjectResource): Source resource with
      POSIX attributes to apply.
    destination_resource (resource_reference.FileObjectResource): Destination
      resource to apply POSIX attributes to.
    known_source_posix (PosixAttributes|None): Use pre-parsed POSIX data instead
      of extracting from source.
    known_destination_posix (PosixAttributes|None): Use pre-parsed POSIX data
      instead of extracting from destination.
    preserve_symlinks (bool): Whether symlinks should be preserved rather than
      followed.

  Raises:
    SystemPermissionError: Custom metadata asked for file ownership change that
      user did not have permission to perform. Other permission errors from
      OS functions are possible. Also see `raise_if_invalid_file_permissions`.
  """
    destination_path = destination_resource.storage_url.object_name
    raise_if_invalid_file_permissions(system_posix_data, source_resource, destination_path, known_posix=known_source_posix)
    custom_posix_attributes = known_source_posix or get_posix_attributes_from_cloud_resource(source_resource)
    existing_posix_attributes = known_destination_posix or get_posix_attributes_from_file(destination_path, preserve_symlinks)
    if custom_posix_attributes.atime is None:
        atime = existing_posix_attributes.atime
        need_utime_call = False
    else:
        atime = custom_posix_attributes.atime
        need_utime_call = custom_posix_attributes.atime != existing_posix_attributes.atime
    if custom_posix_attributes.mtime is None:
        mtime = existing_posix_attributes.mtime
    else:
        mtime = custom_posix_attributes.mtime
        need_utime_call = need_utime_call or custom_posix_attributes.mtime != existing_posix_attributes.mtime
    if need_utime_call:
        follow_symlinks = not preserve_symlinks or os.utime not in os.supports_follow_symlinks
        os.utime(destination_path, (atime, mtime), follow_symlinks=follow_symlinks)
    if platforms.OperatingSystem.IsWindows():
        return
    if custom_posix_attributes.uid is None:
        uid = existing_posix_attributes.uid
        need_chown_call = False
    else:
        uid = custom_posix_attributes.uid
        need_chown_call = custom_posix_attributes.uid != existing_posix_attributes.uid
        if uid != existing_posix_attributes.uid and os.geteuid() != 0:
            os.remove(destination_path)
            raise errors.SystemPermissionError('Root permissions required to set UID {}.'.format(uid))
    if custom_posix_attributes.gid is None:
        gid = existing_posix_attributes.gid
    else:
        gid = custom_posix_attributes.gid
        need_chown_call = need_chown_call or custom_posix_attributes.gid != existing_posix_attributes.gid
    if need_chown_call:
        follow_symlinks = not preserve_symlinks or os.chown not in os.supports_follow_symlinks
        os.chown(destination_path, uid, gid, follow_symlinks=follow_symlinks)
    if custom_posix_attributes.mode is not None and custom_posix_attributes.mode.base_ten_int != existing_posix_attributes.mode.base_ten_int:
        follow_symlinks = not preserve_symlinks or os.chmod not in os.supports_follow_symlinks
        os.chmod(destination_path, custom_posix_attributes.mode.base_ten_int, follow_symlinks=follow_symlinks)