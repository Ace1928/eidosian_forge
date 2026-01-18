from fontTools.misc import psCharStrings
from fontTools import ttLib
from fontTools.pens.basePen import NullPen
from fontTools.misc.roundTools import otRound
from fontTools.misc.loggingTools import deprecateFunction
from fontTools.subset.util import _add_method, _uniq_sort
def processSubr(self, index, subr):
    cs = self.callingStack[-1]
    hints = cs._hints
    subr_hints = subr._hints
    if hints.status != 2:
        for i in range(hints.last_checked, index - 1):
            if isinstance(cs.program[i], str):
                hints.status = 2
                break
        hints.last_checked = index
    if hints.status != 2:
        if subr_hints.has_hint:
            hints.has_hint = True
            if subr_hints.status == 0:
                hints.last_hint = index
            else:
                hints.last_hint = index - 2
    elif subr_hints.status == 0:
        hints.deletions.append(index)
    hints.status = max(hints.status, subr_hints.status)