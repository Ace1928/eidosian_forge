import collections.abc
import functools
import re
import sys
import warnings
from .._utils import set_module
import numpy as np
import numpy.core.numeric as _nx
from numpy.core import transpose
from numpy.core.numeric import (
from numpy.core.umath import (
from numpy.core.fromnumeric import (
from numpy.core.numerictypes import typecodes
from numpy.core import overrides
from numpy.core.function_base import add_newdoc
from numpy.lib.twodim_base import diag
from numpy.core.multiarray import (
from numpy.core.umath import _add_newdoc_ufunc as add_newdoc_ufunc
import builtins
from numpy.lib.histograms import histogram, histogramdd  # noqa: F401
def disp(mesg, device=None, linefeed=True):
    """
    Display a message on a device.

    Parameters
    ----------
    mesg : str
        Message to display.
    device : object
        Device to write message. If None, defaults to ``sys.stdout`` which is
        very similar to ``print``. `device` needs to have ``write()`` and
        ``flush()`` methods.
    linefeed : bool, optional
        Option whether to print a line feed or not. Defaults to True.

    Raises
    ------
    AttributeError
        If `device` does not have a ``write()`` or ``flush()`` method.

    Examples
    --------
    Besides ``sys.stdout``, a file-like object can also be used as it has
    both required methods:

    >>> from io import StringIO
    >>> buf = StringIO()
    >>> np.disp(u'"Display" in a file', device=buf)
    >>> buf.getvalue()
    '"Display" in a file\\n'

    """
    if device is None:
        device = sys.stdout
    if linefeed:
        device.write('%s\n' % mesg)
    else:
        device.write('%s' % mesg)
    device.flush()
    return