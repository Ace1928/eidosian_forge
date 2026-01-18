import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def monthspec(self):
    try:
        try:
            pass
            alt11 = 2
            LA11_0 = self.input.LA(1)
            if LA11_0 == MONTH:
                alt11 = 1
            elif JANUARY <= LA11_0 <= DECEMBER:
                alt11 = 2
            else:
                nvae = NoViableAltException('', 11, 0, self.input)
                raise nvae
            if alt11 == 1:
                pass
                self.match(self.input, MONTH, self.FOLLOW_MONTH_in_monthspec459)
                self.month_set = self.month_set.union(set([self.ValueOf(JANUARY), self.ValueOf(FEBRUARY), self.ValueOf(MARCH), self.ValueOf(APRIL), self.ValueOf(MAY), self.ValueOf(JUNE), self.ValueOf(JULY), self.ValueOf(AUGUST), self.ValueOf(SEPTEMBER), self.ValueOf(OCTOBER), self.ValueOf(NOVEMBER), self.ValueOf(DECEMBER)]))
            elif alt11 == 2:
                pass
                self._state.following.append(self.FOLLOW_months_in_monthspec469)
                self.months()
                self._state.following.pop()
        except RecognitionException as re:
            self.reportError(re)
            self.recover(self.input, re)
    finally:
        pass
    return