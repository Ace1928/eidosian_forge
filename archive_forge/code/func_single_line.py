import os
import stat
import textwrap
from email import message_from_file
from email.message import Message
from tempfile import NamedTemporaryFile
from typing import Optional, List
from distutils.util import rfc822_escape
from . import _normalization, _reqs
from .extern.packaging.markers import Marker
from .extern.packaging.requirements import Requirement
from .extern.packaging.version import Version
from .warnings import SetuptoolsDeprecationWarning
def single_line(val):
    """
    Quick and dirty validation for Summary pypa/setuptools#1390.
    """
    if '\n' in val:
        msg = 'newlines are not allowed in `summary` and will break in the future'
        SetuptoolsDeprecationWarning.emit('Invalid config.', msg)
        val = val.strip().split('\n')[0]
    return val