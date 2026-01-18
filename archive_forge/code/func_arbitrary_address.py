import io
import os
import sys
import socket
import struct
import time
import tempfile
import itertools
from . import util
from . import AuthenticationError, BufferTooShort
from .context import reduction
def arbitrary_address(family):
    """
    Return an arbitrary free address for the given family
    """
    if family == 'AF_INET':
        return ('localhost', 0)
    elif family == 'AF_UNIX':
        return tempfile.mktemp(prefix='listener-', dir=util.get_temp_dir())
    elif family == 'AF_PIPE':
        return tempfile.mktemp(prefix='\\\\.\\pipe\\pyc-%d-%d-' % (os.getpid(), next(_mmap_counter)), dir='')
    else:
        raise ValueError('unrecognized family')