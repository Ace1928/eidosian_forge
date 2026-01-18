import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def quarterspec(self):
    try:
        try:
            pass
            alt13 = 2
            LA13_0 = self.input.LA(1)
            if LA13_0 == QUARTER:
                alt13 = 1
            elif FIRST <= LA13_0 <= THIRD:
                alt13 = 2
            else:
                nvae = NoViableAltException('', 13, 0, self.input)
                raise nvae
            if alt13 == 1:
                pass
                self.match(self.input, QUARTER, self.FOLLOW_QUARTER_in_quarterspec583)
                self.month_set = self.month_set.union(set([self.ValueOf(JANUARY), self.ValueOf(APRIL), self.ValueOf(JULY), self.ValueOf(OCTOBER)]))
            elif alt13 == 2:
                pass
                pass
                self._state.following.append(self.FOLLOW_quarter_ordinals_in_quarterspec595)
                self.quarter_ordinals()
                self._state.following.pop()
                self.match(self.input, MONTH, self.FOLLOW_MONTH_in_quarterspec597)
                self.match(self.input, OF, self.FOLLOW_OF_in_quarterspec599)
                self.match(self.input, QUARTER, self.FOLLOW_QUARTER_in_quarterspec601)
        except RecognitionException as re:
            self.reportError(re)
            self.recover(self.input, re)
    finally:
        pass
    return