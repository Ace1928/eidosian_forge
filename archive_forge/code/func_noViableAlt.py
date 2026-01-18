from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from antlr3.constants import EOF
from antlr3.exceptions import NoViableAltException, BacktrackingFailed
from six.moves import range
def noViableAlt(self, s, input):
    if self.recognizer._state.backtracking > 0:
        raise BacktrackingFailed
    nvae = NoViableAltException(self.getDescription(), self.decisionNumber, s, input)
    self.error(nvae)
    raise nvae