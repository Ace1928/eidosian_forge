from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes, tostr
from fontTools.misc import eexec
from .psOperators import (
import re
from collections.abc import Callable
from string import whitespace
import logging
def do_special(self, token):
    if token == '{':
        self.proclevel = self.proclevel + 1
        return self.procmark
    elif token == '}':
        proc = []
        while 1:
            topobject = self.pop()
            if topobject == self.procmark:
                break
            proc.append(topobject)
        self.proclevel = self.proclevel - 1
        proc.reverse()
        return ps_procedure(proc)
    elif token == '[':
        return self.mark
    elif token == ']':
        return ps_name(']')
    else:
        raise PSTokenError('huh?')