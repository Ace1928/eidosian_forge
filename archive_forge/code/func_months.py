import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def months(self):
    try:
        try:
            pass
            pass
            self._state.following.append(self.FOLLOW_month_in_months486)
            self.month()
            self._state.following.pop()
            while True:
                alt12 = 2
                LA12_0 = self.input.LA(1)
                if LA12_0 == COMMA:
                    alt12 = 1
                if alt12 == 1:
                    pass
                    self.match(self.input, COMMA, self.FOLLOW_COMMA_in_months489)
                    self._state.following.append(self.FOLLOW_month_in_months491)
                    self.month()
                    self._state.following.pop()
                else:
                    break
        except RecognitionException as re:
            self.reportError(re)
            self.recover(self.input, re)
    finally:
        pass
    return