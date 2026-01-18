import base64
import functools
import inspect
import json
import logging
import os
import warnings
import six
from six.moves import urllib
def parse_unique_urlencoded(content):
    """Parses unique key-value parameters from urlencoded content.

    Args:
        content: string, URL-encoded key-value pairs.

    Returns:
        dict, The key-value pairs from ``content``.

    Raises:
        ValueError: if one of the keys is repeated.
    """
    urlencoded_params = urllib.parse.parse_qs(content)
    params = {}
    for key, value in six.iteritems(urlencoded_params):
        if len(value) != 1:
            msg = 'URL-encoded content contains a repeated value:%s -> %s' % (key, ', '.join(value))
            raise ValueError(msg)
        params[key] = value[0]
    return params