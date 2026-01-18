import os
import sys
import typing
from contextlib import contextmanager
from collections.abc import Iterable
from IPython import get_ipython
from traitlets import (
from json import loads as jsonloads, dumps as jsondumps
from .. import comm
from base64 import standard_b64encode
from .utils import deprecation, _get_frame
from .._version import __protocol_version__, __control_protocol_version__, __jupyter_widgets_base_version__
import inspect
@classmethod
def handle_control_comm_opened(cls, comm, msg):
    """
        Class method, called when the comm-open message on the
        "jupyter.widget.control" comm channel is received
        """
    version = msg.get('metadata', {}).get('version', '')
    if version.split('.')[0] != CONTROL_PROTOCOL_VERSION_MAJOR:
        raise ValueError('Incompatible widget control protocol versions: received version %r, expected version %r' % (version, __control_protocol_version__))
    cls._control_comm = comm
    cls._control_comm.on_msg(cls._handle_control_comm_msg)