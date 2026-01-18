import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def mWEDNESDAY(self):
    try:
        _type = WEDNESDAY
        _channel = DEFAULT_CHANNEL
        pass
        self.match('wed')
        alt8 = 2
        LA8_0 = self.input.LA(1)
        if LA8_0 == 110:
            alt8 = 1
        if alt8 == 1:
            pass
            self.match('nesday')
        self._state.type = _type
        self._state.channel = _channel
    finally:
        pass