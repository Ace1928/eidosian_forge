from __future__ import absolute_import
import functools
import os
import socket
import threading
import warnings
def i_am_locking(self):
    """
        Return True if this object is locking the file.
        """
    raise NotImplemented('implement in subclass')