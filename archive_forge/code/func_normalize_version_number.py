import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
def normalize_version_number(version):
    """Turn a version representation into a tuple.

    Examples:

    The following all produce a return value of (1, 0)::

      1, '1', 'v1', [1], (1,), ['1'], 1.0, '1.0', 'v1.0', (1, 0)

    The following all produce a return value of (1, 20, 3)::

      'v1.20.3', '1.20.3', (1, 20, 3), ['1', '20', '3']

    The following all produce a return value of (LATEST, LATEST)::

      'latest', 'vlatest', ('latest', 'latest'), (LATEST, LATEST)

    The following all produce a return value of (2, LATEST)::

      '2.latest', 'v2.latest', (2, LATEST), ('2', 'latest')

    :param version: A version specifier in any of the following forms:
        String, possibly prefixed with 'v', containing one or more numbers
        *or* the string 'latest', separated by periods.  Examples: 'v1',
        'v1.2', '1.2.3', '123', 'latest', '1.latest', 'v1.latest'.
        Integer.  This will be assumed to be the major version, with a minor
        version of 0.
        Float.  The integer part is assumed to be the major version; the
        decimal part the minor version.
        Non-string iterable comprising integers, integer strings, the string
        'latest', or the special value LATEST.
        Examples: (1,), [1, 2], ('12', '34', '56'), (LATEST,), (2, 'latest')
    :return: A tuple of len >= 2 comprising integers and/or LATEST.
    :raises TypeError: If the input version cannot be interpreted.
    """
    ver = version
    if not isinstance(ver, str):
        try:
            ver = '.'.join(map(_str_or_latest, ver))
        except TypeError:
            pass
    if isinstance(ver, str):
        ver = ver.lstrip('v')
        try:
            ver = str(float(int(ver)))
        except ValueError:
            pass
    elif isinstance(ver, (int, float)):
        ver = _str_or_latest(float(ver))
    try:
        ver = ver.split('.')
    except AttributeError:
        pass
    if ver == 'latest' or tuple(ver) == ('latest',):
        return (LATEST, LATEST)
    try:
        return tuple(map(_int_or_latest, ver))
    except (TypeError, ValueError):
        pass
    raise TypeError('Invalid version specified: %s' % version)