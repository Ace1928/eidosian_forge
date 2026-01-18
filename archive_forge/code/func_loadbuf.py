import functools
import io
import os
import pickle
import socket
import sys
from . import context
@classmethod
def loadbuf(cls, buf, protocol=None):
    return cls.loads(buf.getvalue())