import atexit
from collections import namedtuple
from collections.abc import MutableMapping
import contextlib
import functools
import importlib
import inspect
from inspect import Parameter
import locale
import logging
import os
from pathlib import Path
import pprint
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
import numpy
from packaging.version import parse as parse_version
from . import _api, _version, cbook, _docstring, rcsetup
from matplotlib.cbook import sanitize_sequence
from matplotlib._api import MatplotlibDeprecationWarning
from matplotlib.rcsetup import validate_backend, cycler
from matplotlib.cm import _colormaps as colormaps
from matplotlib.colors import _color_sequences as color_sequences
@contextlib.contextmanager
def rc_context(rc=None, fname=None):
    """
    Return a context manager for temporarily changing rcParams.

    The :rc:`backend` will not be reset by the context manager.

    rcParams changed both through the context manager invocation and
    in the body of the context will be reset on context exit.

    Parameters
    ----------
    rc : dict
        The rcParams to temporarily set.
    fname : str or path-like
        A file with Matplotlib rc settings. If both *fname* and *rc* are given,
        settings from *rc* take precedence.

    See Also
    --------
    :ref:`customizing-with-matplotlibrc-files`

    Examples
    --------
    Passing explicit values via a dict::

        with mpl.rc_context({'interactive': False}):
            fig, ax = plt.subplots()
            ax.plot(range(3), range(3))
            fig.savefig('example.png')
            plt.close(fig)

    Loading settings from a file::

         with mpl.rc_context(fname='print.rc'):
             plt.plot(x, y)  # uses 'print.rc'

    Setting in the context body::

        with mpl.rc_context():
            # will be reset
            mpl.rcParams['lines.linewidth'] = 5
            plt.plot(x, y)

    """
    orig = dict(rcParams.copy())
    del orig['backend']
    try:
        if fname:
            rc_file(fname)
        if rc:
            rcParams.update(rc)
        yield
    finally:
        dict.update(rcParams, orig)