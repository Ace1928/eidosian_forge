import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def mMINUTES(self):
    try:
        _type = MINUTES
        _channel = DEFAULT_CHANNEL
        pass
        alt24 = 2
        LA24_0 = self.input.LA(1)
        if LA24_0 == 109:
            LA24_1 = self.input.LA(2)
            if LA24_1 == 105:
                LA24_2 = self.input.LA(3)
                if LA24_2 == 110:
                    LA24_3 = self.input.LA(4)
                    if LA24_3 == 115:
                        alt24 = 1
                    elif LA24_3 == 117:
                        alt24 = 2
                    else:
                        if self._state.backtracking > 0:
                            raise BacktrackingFailed
                        nvae = NoViableAltException('', 24, 3, self.input)
                        raise nvae
                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed
                    nvae = NoViableAltException('', 24, 2, self.input)
                    raise nvae
            else:
                if self._state.backtracking > 0:
                    raise BacktrackingFailed
                nvae = NoViableAltException('', 24, 1, self.input)
                raise nvae
        else:
            if self._state.backtracking > 0:
                raise BacktrackingFailed
            nvae = NoViableAltException('', 24, 0, self.input)
            raise nvae
        if alt24 == 1:
            pass
            self.match('mins')
        elif alt24 == 2:
            pass
            self.match('minutes')
        self._state.type = _type
        self._state.channel = _channel
    finally:
        pass