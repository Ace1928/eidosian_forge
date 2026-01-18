from fontTools.misc import psCharStrings
from fontTools import ttLib
from fontTools.pens.basePen import NullPen
from fontTools.misc.roundTools import otRound
from fontTools.misc.loggingTools import deprecateFunction
from fontTools.subset.util import _add_method, _uniq_sort
def processHintmask(self, index):
    cs = self.callingStack[-1]
    hints = cs._hints
    hints.has_hintmask = True
    if hints.status != 2:
        for i in range(hints.last_checked, index - 1):
            if isinstance(cs.program[i], str):
                hints.status = 2
                break
        else:
            hints.has_hint = True
            hints.last_hint = index + 1
            hints.status = 0
    hints.last_checked = index + 1