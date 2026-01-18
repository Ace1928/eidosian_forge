from fontTools.misc import psCharStrings
from fontTools import ttLib
from fontTools.pens.basePen import NullPen
from fontTools.misc.roundTools import otRound
from fontTools.misc.loggingTools import deprecateFunction
from fontTools.subset.util import _add_method, _uniq_sort
@_add_method(psCharStrings.T2CharString)
def subset_subroutines(self, subrs, gsubrs):
    p = self.program
    for i in range(1, len(p)):
        if p[i] == 'callsubr':
            assert isinstance(p[i - 1], int)
            p[i - 1] = subrs._used.index(p[i - 1] + subrs._old_bias) - subrs._new_bias
        elif p[i] == 'callgsubr':
            assert isinstance(p[i - 1], int)
            p[i - 1] = gsubrs._used.index(p[i - 1] + gsubrs._old_bias) - gsubrs._new_bias