from fontTools.misc import psCharStrings
from fontTools import ttLib
from fontTools.pens.basePen import NullPen
from fontTools.misc.roundTools import otRound
from fontTools.misc.loggingTools import deprecateFunction
from fontTools.subset.util import _add_method, _uniq_sort
@_add_method(psCharStrings.T2CharString)
def drop_hints(self):
    hints = self._hints
    if hints.deletions:
        p = self.program
        for idx in reversed(hints.deletions):
            del p[idx - 2:idx]
    if hints.has_hint:
        assert not hints.deletions or hints.last_hint <= hints.deletions[0]
        self.program = self.program[hints.last_hint:]
        if not self.program:
            self.program.append('endchar')
        if hasattr(self, 'width'):
            if self.width != self.private.defaultWidthX:
                assert self.private.defaultWidthX is not None, 'CFF2 CharStrings must not have an initial width value'
                self.program.insert(0, self.width - self.private.nominalWidthX)
    if hints.has_hintmask:
        i = 0
        p = self.program
        while i < len(p):
            if p[i] in ['hintmask', 'cntrmask']:
                assert i + 1 <= len(p)
                del p[i:i + 2]
                continue
            i += 1
    assert len(self.program)
    del self._hints