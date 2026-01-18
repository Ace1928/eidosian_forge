import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def mWS(self):
    try:
        _type = WS
        _channel = DEFAULT_CHANNEL
        pass
        if 9 <= self.input.LA(1) <= 10 or self.input.LA(1) == 13 or self.input.LA(1) == 32:
            self.input.consume()
        else:
            if self._state.backtracking > 0:
                raise BacktrackingFailed
            mse = MismatchedSetException(None, self.input)
            self.recover(mse)
            raise mse
        if self._state.backtracking == 0:
            _channel = HIDDEN
        self._state.type = _type
        self._state.channel = _channel
    finally:
        pass