import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def mDIGITS(self):
    try:
        _type = DIGITS
        _channel = DEFAULT_CHANNEL
        pass
        alt25 = 4
        LA25_0 = self.input.LA(1)
        if 48 <= LA25_0 <= 57:
            LA25_1 = self.input.LA(2)
            if 48 <= LA25_1 <= 57:
                LA25_2 = self.input.LA(3)
                if 48 <= LA25_2 <= 57:
                    LA25_4 = self.input.LA(4)
                    if 48 <= LA25_4 <= 57:
                        LA25_6 = self.input.LA(5)
                        if 48 <= LA25_6 <= 57 and self.synpred1_Groc():
                            alt25 = 1
                        else:
                            alt25 = 2
                    else:
                        alt25 = 3
                else:
                    alt25 = 4
            else:
                if self._state.backtracking > 0:
                    raise BacktrackingFailed
                nvae = NoViableAltException('', 25, 1, self.input)
                raise nvae
        else:
            if self._state.backtracking > 0:
                raise BacktrackingFailed
            nvae = NoViableAltException('', 25, 0, self.input)
            raise nvae
        if alt25 == 1:
            pass
            pass
            self.mDIGIT()
            self.mDIGIT()
            self.mDIGIT()
            self.mDIGIT()
            self.mDIGIT()
        elif alt25 == 2:
            pass
            pass
            self.mDIGIT()
            self.mDIGIT()
            self.mDIGIT()
            self.mDIGIT()
        elif alt25 == 3:
            pass
            pass
            self.mDIGIT()
            self.mDIGIT()
            self.mDIGIT()
        elif alt25 == 4:
            pass
            pass
            self.mDIGIT()
            self.mDIGIT()
        self._state.type = _type
        self._state.channel = _channel
    finally:
        pass