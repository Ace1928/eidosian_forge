import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def mSUNDAY(self):
    try:
        _type = SUNDAY
        _channel = DEFAULT_CHANNEL
        pass
        self.match('sun')
        alt12 = 2
        LA12_0 = self.input.LA(1)
        if LA12_0 == 100:
            alt12 = 1
        if alt12 == 1:
            pass
            self.match('day')
        self._state.type = _type
        self._state.channel = _channel
    finally:
        pass