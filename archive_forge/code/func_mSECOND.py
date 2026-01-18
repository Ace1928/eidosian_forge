import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def mSECOND(self):
    try:
        _type = SECOND
        _channel = DEFAULT_CHANNEL
        pass
        alt3 = 2
        LA3_0 = self.input.LA(1)
        if LA3_0 == 50:
            alt3 = 1
        elif LA3_0 == 115:
            alt3 = 2
        else:
            if self._state.backtracking > 0:
                raise BacktrackingFailed
            nvae = NoViableAltException('', 3, 0, self.input)
            raise nvae
        if alt3 == 1:
            pass
            self.match('2nd')
        elif alt3 == 2:
            pass
            self.match('second')
        self._state.type = _type
        self._state.channel = _channel
    finally:
        pass