import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def mTHIRD(self):
    try:
        _type = THIRD
        _channel = DEFAULT_CHANNEL
        pass
        alt4 = 2
        LA4_0 = self.input.LA(1)
        if LA4_0 == 51:
            alt4 = 1
        elif LA4_0 == 116:
            alt4 = 2
        else:
            if self._state.backtracking > 0:
                raise BacktrackingFailed
            nvae = NoViableAltException('', 4, 0, self.input)
            raise nvae
        if alt4 == 1:
            pass
            self.match('3rd')
        elif alt4 == 2:
            pass
            self.match('third')
        self._state.type = _type
        self._state.channel = _channel
    finally:
        pass