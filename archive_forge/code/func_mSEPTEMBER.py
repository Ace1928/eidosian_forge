import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def mSEPTEMBER(self):
    try:
        _type = SEPTEMBER
        _channel = DEFAULT_CHANNEL
        pass
        self.match('sep')
        alt20 = 2
        LA20_0 = self.input.LA(1)
        if LA20_0 == 116:
            alt20 = 1
        if alt20 == 1:
            pass
            self.match('tember')
        self._state.type = _type
        self._state.channel = _channel
    finally:
        pass