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
def rebuild_connection(df, readable, writable):
    fd = df.detach()
    return Connection(fd, readable, writable)