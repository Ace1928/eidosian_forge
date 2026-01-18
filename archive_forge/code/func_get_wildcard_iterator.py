from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import fnmatch
import heapq
import os
import pathlib
import re
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import errors as api_errors
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.command_lib.storage import errors as command_errors
from googlecloudsdk.command_lib.storage import folder_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import log
from googlecloudsdk.core.util import debug_output
import six
def get_wildcard_iterator(url_str, error_on_missing_key=True, exclude_patterns=None, fetch_encrypted_object_hashes=False, fields_scope=cloud_api.FieldsScope.NO_ACL, files_only=False, force_include_hidden_files=False, get_bucket_metadata=False, halt_on_empty_response=True, ignore_symlinks=False, managed_folder_setting=folder_util.ManagedFolderSetting.DO_NOT_LIST, next_page_token=None, object_state=cloud_api.ObjectState.LIVE, preserve_symlinks=False, raise_managed_folder_precondition_errors=False):
    """Instantiate a WildcardIterator for the given URL string.

  Args:
    url_str (str): URL string which may contain wildcard characters.
    error_on_missing_key (bool): If true, and the encryption key needed to
      decrypt an object is missing, the iterator raises an error for that
      object.
    exclude_patterns (Patterns|None): Don't return resources whose URLs or local
      file paths matched these regex patterns.
    fetch_encrypted_object_hashes (bool): Fall back to GET requests for
      encrypted cloud objects in order to fetch their hash values.
    fields_scope (cloud_api.FieldsScope): Determines amount of metadata returned
      by API.
    files_only (bool): Skips containers. Raises error for stream types. Still
      returns symlinks.
    force_include_hidden_files (bool): Include local hidden files even if not
      recursive iteration. URL should be for directory or directory followed by
      wildcards.
    get_bucket_metadata (bool): If true, perform a bucket GET request when
      fetching bucket resources.
    halt_on_empty_response (bool): Stops querying after empty list response. See
      CloudApi for details.
    ignore_symlinks (bool): Skip over symlinks instead of following them.
    managed_folder_setting (folder_util.ManagedFolderSetting): Indicates how to
      deal with managed folders.
    next_page_token (str|None): Used to resume LIST calls.
    object_state (cloud_api.ObjectState): Versions of objects to query.
    preserve_symlinks (bool): Preserve symlinks instead of following them.
    raise_managed_folder_precondition_errors (bool): If True, raises
      precondition errors from managed folder listing. Otherwise, suppresses
      these errors. This is helpful in commands that list managed folders by
      default.

  Returns:
    A WildcardIterator object.
  """
    url = storage_url.storage_url_from_string(url_str)
    if isinstance(url, storage_url.CloudUrl):
        return CloudWildcardIterator(url, error_on_missing_key=error_on_missing_key, exclude_patterns=exclude_patterns, fetch_encrypted_object_hashes=fetch_encrypted_object_hashes, fields_scope=fields_scope, files_only=files_only, get_bucket_metadata=get_bucket_metadata, halt_on_empty_response=halt_on_empty_response, managed_folder_setting=managed_folder_setting, next_page_token=next_page_token, object_state=object_state, raise_managed_folder_precondition_errors=raise_managed_folder_precondition_errors)
    elif isinstance(url, storage_url.FileUrl):
        return FileWildcardIterator(url, exclude_patterns=exclude_patterns, files_only=files_only, force_include_hidden_files=force_include_hidden_files, ignore_symlinks=ignore_symlinks, preserve_symlinks=preserve_symlinks)
    else:
        raise command_errors.InvalidUrlError('Unknown url type %s.' % url)