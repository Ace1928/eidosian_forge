import datetime
import numbers
import re
import sys
import os
import textwrap
from tornado.escape import _unicode, native_str
from tornado.log import define_logging_options
from tornado.util import basestring_type, exec_in
from typing import (
def mockable(self) -> '_Mockable':
    """Returns a wrapper around self that is compatible with
        `mock.patch <unittest.mock.patch>`.

        The `mock.patch <unittest.mock.patch>` function (included in
        the standard library `unittest.mock` package since Python 3.3,
        or in the third-party ``mock`` package for older versions of
        Python) is incompatible with objects like ``options`` that
        override ``__getattr__`` and ``__setattr__``.  This function
        returns an object that can be used with `mock.patch.object
        <unittest.mock.patch.object>` to modify option values::

            with mock.patch.object(options.mockable(), 'name', value):
                assert options.name == value
        """
    return _Mockable(self)