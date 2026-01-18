import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def mFOURTH_OR_FIFTH(self):
    try:
        _type = FOURTH_OR_FIFTH
        _channel = DEFAULT_CHANNEL
        pass
        alt5 = 2
        LA5_0 = self.input.LA(1)
        if LA5_0 == 102:
            LA5_1 = self.input.LA(2)
            if LA5_1 == 111:
                alt5 = 1
            elif LA5_1 == 105:
                alt5 = 2
            else:
                if self._state.backtracking > 0:
                    raise BacktrackingFailed
                nvae = NoViableAltException('', 5, 1, self.input)
                raise nvae
        else:
            if self._state.backtracking > 0:
                raise BacktrackingFailed
            nvae = NoViableAltException('', 5, 0, self.input)
            raise nvae
        if alt5 == 1:
            pass
            pass
            self.match('fourth')
            if self._state.backtracking == 0:
                _type = FOURTH
        elif alt5 == 2:
            pass
            pass
            self.match('fifth')
            if self._state.backtracking == 0:
                _type = FIFTH
        self._state.type = _type
        self._state.channel = _channel
    finally:
        pass