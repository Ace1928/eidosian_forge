from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import os
import re
import subprocess
from boto import config
from gslib import exception
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.utils import boto_util
from gslib.utils import constants
from gslib.utils import system_util
def get_flag_from_header(raw_header_key, header_value, unset=False):
    """Returns the gcloud storage flag for the given gsutil header.

  Args:
    raw_header_key: The header key.
    header_value: The header value
    unset: If True, the equivalent clear/remove flag is returned instead of the
      setter flag. This only applies to setmeta.

  Returns:
    A string representing the equivalent gcloud storage flag and value, if
      translation is possible, else returns None.

  Examples:
    >> get_flag_from_header('Cache-Control', 'val')
    --cache-control=val

    >> get_flag_from_header('x-goog-meta-foo', 'val')
    --update-custom-metadata=foo=val

    >> get_flag_from_header('x-goog-meta-foo', 'val', unset=True)
    --remove-custom-metadata=foo

  """
    lowercase_header_key = raw_header_key.lower()
    if lowercase_header_key in PRECONDITIONS_HEADERS:
        providerless_flag = raw_header_key[len('x-goog-'):]
        if not providerless_flag.startswith('if-'):
            flag_name = 'if-' + providerless_flag
        else:
            flag_name = providerless_flag
    elif lowercase_header_key in DATA_TRANSFER_HEADERS:
        flag_name = lowercase_header_key
    else:
        flag_name = None
    if flag_name is not None:
        if unset:
            if lowercase_header_key in PRECONDITIONS_HEADERS or lowercase_header_key == 'content-md5':
                return None
            else:
                return '--clear-' + flag_name
        return '--{}={}'.format(flag_name, header_value)
    for header_prefix in ('x-goog-meta-', 'x-amz-meta-'):
        if lowercase_header_key.startswith(header_prefix):
            metadata_key = raw_header_key[len(header_prefix):]
            if unset:
                return '--remove-custom-metadata=' + metadata_key
            else:
                return '--update-custom-metadata={}={}'.format(metadata_key, header_value)
    return None