import functools
import re
import packaging.version
from oslo_utils._i18n import _
def convert_version_to_int(version):
    """Convert a version to an integer.

    *version* must be a string with dots or a tuple of integers.

    .. versionadded:: 2.0
    """
    try:
        if isinstance(version, str):
            version = convert_version_to_tuple(version)
        if isinstance(version, tuple):
            return functools.reduce(lambda x, y: x * 1000 + y, version)
    except Exception as ex:
        msg = _('Version %s is invalid.') % version
        raise ValueError(msg) from ex