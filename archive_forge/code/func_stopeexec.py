from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes, tostr
from fontTools.misc import eexec
from .psOperators import (
import re
from collections.abc import Callable
from string import whitespace
import logging
def stopeexec(self):
    if not hasattr(self, 'dirtybuf'):
        return
    self.buf = self.dirtybuf
    del self.dirtybuf