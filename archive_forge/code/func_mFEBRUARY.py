import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def mFEBRUARY(self):
    try:
        _type = FEBRUARY
        _channel = DEFAULT_CHANNEL
        pass
        self.match('feb')
        alt14 = 2
        LA14_0 = self.input.LA(1)
        if LA14_0 == 114:
            alt14 = 1
        if alt14 == 1:
            pass
            self.match('ruary')
        self._state.type = _type
        self._state.channel = _channel
    finally:
        pass