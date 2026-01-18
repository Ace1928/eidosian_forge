import array
import os
import struct
import six
from ._exceptions import *
from ._utils import validate_utf8
from threading import Lock
def recv_mask(self):
    self.mask = self.recv_strict(4) if self.has_mask() else ''