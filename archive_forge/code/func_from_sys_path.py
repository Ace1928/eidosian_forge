import collections
import os
import re
import sys
from datetime import timedelta
from pyudev._errors import (
from pyudev._util import (
@classmethod
def from_sys_path(cls, context, sys_path):
    """
        .. versionchanged:: 0.4
           Raise :exc:`NoSuchDeviceError` instead of returning ``None``, if
           no device was found for ``sys_path``.
        .. versionchanged:: 0.5
           Raise :exc:`DeviceNotFoundAtPathError` instead of
           :exc:`NoSuchDeviceError`.
        .. deprecated:: 0.18
           Use :class:`Devices.from_sys_path` instead.
        """
    import warnings
    warnings.warn('Will be removed in 1.0. Use equivalent Devices method instead.', DeprecationWarning, stacklevel=2)
    return Devices.from_sys_path(context, sys_path)