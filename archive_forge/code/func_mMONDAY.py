import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def mMONDAY(self):
    try:
        _type = MONDAY
        _channel = DEFAULT_CHANNEL
        pass
        self.match('mon')
        alt6 = 2
        LA6_0 = self.input.LA(1)
        if LA6_0 == 100:
            alt6 = 1
        if alt6 == 1:
            pass
            self.match('day')
        self._state.type = _type
        self._state.channel = _channel
    finally:
        pass