from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import os
import re
import stat
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
import six
from six.moves import urllib
def storage_url_from_string(url_string):
    """Static factory function for creating a StorageUrl from a string.

  Args:
    url_string (str): Cloud url or local filepath.

  Returns:
     StorageUrl object.

  Raises:
    InvalidUrlError: Unrecognized URL scheme.
  """
    scheme = _get_scheme_from_url_string(url_string)
    if scheme == ProviderPrefix.FILE:
        return FileUrl(url_string)
    if scheme == ProviderPrefix.POSIX:
        return PosixFileSystemUrl(url_string)
    if scheme == ProviderPrefix.HDFS:
        return HdfsUrl(url_string)
    if scheme in VALID_HTTP_SCHEMES:
        return AzureUrl.from_url_string(url_string)
    if scheme in VALID_CLOUD_SCHEMES:
        return CloudUrl.from_url_string(url_string)
    raise errors.InvalidUrlError('Unrecognized URL scheme.')