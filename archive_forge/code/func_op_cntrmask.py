from fontTools.misc import psCharStrings
from fontTools import ttLib
from fontTools.pens.basePen import NullPen
from fontTools.misc.roundTools import otRound
from fontTools.misc.loggingTools import deprecateFunction
from fontTools.subset.util import _add_method, _uniq_sort
def op_cntrmask(self, index):
    rv = psCharStrings.T2WidthExtractor.op_cntrmask(self, index)
    self.processHintmask(index)
    return rv