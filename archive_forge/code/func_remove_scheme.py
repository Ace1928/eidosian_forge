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
def remove_scheme(url_string):
    """Removes ProviderPrefix or other scheme from URL string."""
    if SCHEME_DELIMITER not in url_string:
        return url_string
    _, _, schemeless_url = url_string.partition(SCHEME_DELIMITER)
    return schemeless_url