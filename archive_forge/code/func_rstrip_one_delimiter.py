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
def rstrip_one_delimiter(string, delimiter=CloudUrl.CLOUD_URL_DELIM):
    """Strip one delimiter char from the end.

  Args:
    string (str): String on which the action needs to be performed.
    delimiter (str): A delimiter char.

  Returns:
    str: String with trailing delimiter removed.
  """
    if string.endswith(delimiter):
        return string[:-len(delimiter)]
    return string