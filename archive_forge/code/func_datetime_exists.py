from relative deltas), local machine timezone, fixed offset timezone, and UTC
import datetime
import logging  # GOOGLE
import struct
import time
import sys
import os
import bisect
import weakref
from collections import OrderedDict
import six
from six import string_types
from six.moves import _thread
from ._common import tzname_in_python2, _tzinfo
from ._common import tzrangebase, enfold
from ._common import _validate_fromutc_inputs
from ._factories import _TzSingleton, _TzOffsetFactory
from ._factories import _TzStrFactory
from warnings import warn
def datetime_exists(dt, tz=None):
    """
    Given a datetime and a time zone, determine whether or not a given datetime
    would fall in a gap.

    :param dt:
        A :class:`datetime.datetime` (whose time zone will be ignored if ``tz``
        is provided.)

    :param tz:
        A :class:`datetime.tzinfo` with support for the ``fold`` attribute. If
        ``None`` or not provided, the datetime's own time zone will be used.

    :return:
        Returns a boolean value whether or not the "wall time" exists in
        ``tz``.

    .. versionadded:: 2.7.0
    """
    if tz is None:
        if dt.tzinfo is None:
            raise ValueError('Datetime is naive and no time zone provided.')
        tz = dt.tzinfo
    dt = dt.replace(tzinfo=None)
    dt_rt = dt.replace(tzinfo=tz).astimezone(UTC).astimezone(tz)
    dt_rt = dt_rt.replace(tzinfo=None)
    return dt == dt_rt