from fontTools.misc import psCharStrings
from fontTools import ttLib
from fontTools.pens.basePen import NullPen
from fontTools.misc.roundTools import otRound
from fontTools.misc.loggingTools import deprecateFunction
from fontTools.subset.util import _add_method, _uniq_sort
@_add_method(ttLib.getTableClass('CFF '))
def remove_unused_subroutines(self):
    cff = self.cff
    for fontname in cff.keys():
        font = cff[fontname]
        cs = font.CharStrings
        for g in font.charset:
            c, _ = cs.getItemAndSelector(g)
            subrs = getattr(c.private, 'Subrs', [])
            decompiler = _MarkingT2Decompiler(subrs, c.globalSubrs, c.private)
            decompiler.execute(c)
        all_subrs = [font.GlobalSubrs]
        if hasattr(font, 'FDArray'):
            all_subrs.extend((fd.Private.Subrs for fd in font.FDArray if hasattr(fd.Private, 'Subrs') and fd.Private.Subrs))
        elif hasattr(font.Private, 'Subrs') and font.Private.Subrs:
            all_subrs.append(font.Private.Subrs)
        subrs = set(subrs)
        for subrs in all_subrs:
            if not hasattr(subrs, '_used'):
                subrs._used = set()
            subrs._used = _uniq_sort(subrs._used)
            subrs._old_bias = psCharStrings.calcSubrBias(subrs)
            subrs._new_bias = psCharStrings.calcSubrBias(subrs._used)
        for g in font.charset:
            c, _ = cs.getItemAndSelector(g)
            subrs = getattr(c.private, 'Subrs', None)
            c.subset_subroutines(subrs, font.GlobalSubrs)
        for subrs in all_subrs:
            if subrs == font.GlobalSubrs:
                if not hasattr(font, 'FDArray') and hasattr(font.Private, 'Subrs'):
                    local_subrs = font.Private.Subrs
                else:
                    local_subrs = None
            else:
                local_subrs = subrs
            subrs.items = [subrs.items[i] for i in subrs._used]
            if hasattr(subrs, 'file'):
                del subrs.file
            if hasattr(subrs, 'offsets'):
                del subrs.offsets
            for subr in subrs.items:
                subr.subset_subroutines(local_subrs, font.GlobalSubrs)
        if hasattr(font, 'FDArray'):
            for fd in font.FDArray:
                _delete_empty_subrs(fd.Private)
        else:
            _delete_empty_subrs(font.Private)
        for subrs in all_subrs:
            del subrs._used, subrs._old_bias, subrs._new_bias