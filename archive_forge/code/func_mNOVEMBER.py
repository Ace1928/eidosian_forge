import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def mNOVEMBER(self):
    try:
        _type = NOVEMBER
        _channel = DEFAULT_CHANNEL
        pass
        self.match('nov')
        alt22 = 2
        LA22_0 = self.input.LA(1)
        if LA22_0 == 101:
            alt22 = 1
        if alt22 == 1:
            pass
            self.match('ember')
        self._state.type = _type
        self._state.channel = _channel
    finally:
        pass