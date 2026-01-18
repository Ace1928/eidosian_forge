import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def mSYNCHRONIZED(self):
    try:
        _type = SYNCHRONIZED
        _channel = DEFAULT_CHANNEL
        pass
        self.match('synchronized')
        self._state.type = _type
        self._state.channel = _channel
    finally:
        pass