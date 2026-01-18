import logging
import re
import statsd
import webob.dec
from oslo_middleware import base
@staticmethod
def strip_uuid(path):
    """Remove normal-form UUID from supplied path.

        Only call after replacing slashes with dots in path.
        """
    match = UUID_REGEX.match(path)
    if match is None:
        return path
    return path.replace(match.group(1), '')