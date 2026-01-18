import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def month_of_quarter_ordinal(self):
    offset = None
    try:
        try:
            pass
            offset = self.input.LT(1)
            if FIRST <= self.input.LA(1) <= THIRD:
                self.input.consume()
                self._state.errorRecovery = False
            else:
                mse = MismatchedSetException(None, self.input)
                raise mse
            jOffset = self.ValueOf(offset.type) - 1
            self.month_set = self.month_set.union(set([jOffset + self.ValueOf(JANUARY), jOffset + self.ValueOf(APRIL), jOffset + self.ValueOf(JULY), jOffset + self.ValueOf(OCTOBER)]))
        except RecognitionException as re:
            self.reportError(re)
            self.recover(self.input, re)
    finally:
        pass
    return