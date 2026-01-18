import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
def join_segment_parameters(url, parameters):
    """Create a new URL by adding segment parameters to an existing one.

    The parameters of the last segment in the URL will be updated; if a
    parameter with the same key already exists it will be overwritten.

    Args:
      url: A URL, as string
      parameters: Dictionary of parameters, keys and values as bytestrings
    """
    base, existing_parameters = split_segment_parameters(url)
    new_parameters = {}
    new_parameters.update(existing_parameters)
    for key, value in parameters.items():
        if not isinstance(key, str):
            raise TypeError('parameter key %r is not a str' % key)
        if not isinstance(value, str):
            raise TypeError('parameter value %r for %r is not a str' % (value, key))
        if '=' in key:
            raise InvalidURLJoin('= exists in parameter key', url, parameters)
        new_parameters[key] = value
    return join_segment_parameters_raw(base, *['%s=%s' % item for item in sorted(new_parameters.items())])