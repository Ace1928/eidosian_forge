import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def specifictime(self):
    TIME1 = None
    try:
        try:
            pass
            pass
            alt4 = 2
            alt4 = self.dfa4.predict(self.input)
            if alt4 == 1:
                pass
                pass
                alt2 = 2
                LA2_0 = self.input.LA(1)
                if LA2_0 == EVERY or FIRST <= LA2_0 <= FOURTH_OR_FIFTH:
                    alt2 = 1
                elif DIGIT <= LA2_0 <= DIGITS:
                    alt2 = 2
                else:
                    nvae = NoViableAltException('', 2, 0, self.input)
                    raise nvae
                if alt2 == 1:
                    pass
                    pass
                    self._state.following.append(self.FOLLOW_ordinals_in_specifictime72)
                    self.ordinals()
                    self._state.following.pop()
                    self._state.following.append(self.FOLLOW_weekdays_in_specifictime74)
                    self.weekdays()
                    self._state.following.pop()
                elif alt2 == 2:
                    pass
                    self._state.following.append(self.FOLLOW_monthdays_in_specifictime77)
                    self.monthdays()
                    self._state.following.pop()
                self.match(self.input, OF, self.FOLLOW_OF_in_specifictime80)
                alt3 = 2
                LA3_0 = self.input.LA(1)
                if MONTH <= LA3_0 <= DECEMBER:
                    alt3 = 1
                elif FIRST <= LA3_0 <= THIRD or LA3_0 == QUARTER:
                    alt3 = 2
                else:
                    nvae = NoViableAltException('', 3, 0, self.input)
                    raise nvae
                if alt3 == 1:
                    pass
                    self._state.following.append(self.FOLLOW_monthspec_in_specifictime83)
                    self.monthspec()
                    self._state.following.pop()
                elif alt3 == 2:
                    pass
                    self._state.following.append(self.FOLLOW_quarterspec_in_specifictime85)
                    self.quarterspec()
                    self._state.following.pop()
            elif alt4 == 2:
                pass
                pass
                self._state.following.append(self.FOLLOW_ordinals_in_specifictime101)
                self.ordinals()
                self._state.following.pop()
                self._state.following.append(self.FOLLOW_weekdays_in_specifictime103)
                self.weekdays()
                self._state.following.pop()
                self.month_set = set(range(1, 13))
            TIME1 = self.match(self.input, TIME, self.FOLLOW_TIME_in_specifictime117)
            self.time_string = TIME1.text
        except RecognitionException as re:
            self.reportError(re)
            self.recover(self.input, re)
    finally:
        pass
    return