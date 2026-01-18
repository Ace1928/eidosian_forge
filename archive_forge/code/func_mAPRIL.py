import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def mAPRIL(self):
    try:
        _type = APRIL
        _channel = DEFAULT_CHANNEL
        pass
        self.match('apr')
        alt16 = 2
        LA16_0 = self.input.LA(1)
        if LA16_0 == 105:
            alt16 = 1
        if alt16 == 1:
            pass
            self.match('il')
        self._state.type = _type
        self._state.channel = _channel
    finally:
        pass