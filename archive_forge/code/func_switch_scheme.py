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
def switch_scheme(original_url, new_scheme):
    """Returns best-effort new StorageUrl based on original with new scheme.

  This relies strongly on "storage_url_from_string" and will probably fail
  for unusual formats like Azure URL. However, delimiter replacement is
  handled for cases like converting Windows to cloud URLs.

  Ignores versioning info embedded in URLs because each URL type tends to have
  non-translatable syntax for its versions.

  Args:
    original_url (StorageUrl): URL to convert.
    new_scheme (ProviderPrefix): Scheme to update URL with. probably fail or
      have unexpected results because URL formats tend to have non-translatable
      versioning syntax.

  Returns:
    StorageUrl with updated scheme and best-effort transformation.
  """
    _, old_url_string_no_scheme = original_url.versionless_url_string.split(SCHEME_DELIMITER)
    unprocessed_new_url = storage_url_from_string('{}{}{}'.format(new_scheme.value, SCHEME_DELIMITER, old_url_string_no_scheme))
    if original_url.delimiter == unprocessed_new_url.delimiter:
        return unprocessed_new_url
    old_url_string_no_scheme_correct_delimiter = old_url_string_no_scheme.replace(original_url.delimiter, unprocessed_new_url.delimiter)
    return storage_url_from_string('{}{}{}'.format(new_scheme.value, SCHEME_DELIMITER, old_url_string_no_scheme_correct_delimiter))