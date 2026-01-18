from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes, tostr
from fontTools.misc import eexec
from .psOperators import (
import re
from collections.abc import Callable
from string import whitespace
import logging
def starteexec(self):
    self.pos = self.pos + 1
    self.dirtybuf = self.buf[self.pos:]
    self.buf, R = eexec.decrypt(self.dirtybuf, 55665)
    self.len = len(self.buf)
    self.pos = 4