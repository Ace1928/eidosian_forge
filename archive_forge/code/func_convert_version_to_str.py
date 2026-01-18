import functools
import re
import packaging.version
from oslo_utils._i18n import _
def convert_version_to_str(version_int):
    """Convert a version integer to a string with dots.

    .. versionadded:: 2.0
    """
    version_numbers = []
    factor = 1000
    while version_int != 0:
        version_number = version_int - version_int // factor * factor
        version_numbers.insert(0, str(version_number))
        version_int = version_int // factor
    return '.'.join(map(str, version_numbers))