from collections import namedtuple
from fontTools.cffLib import (
from io import BytesIO
from fontTools.cffLib.specializer import specializeCommands, commandsToProgram
from fontTools.ttLib import newTable
from fontTools import varLib
from fontTools.varLib.models import allEqual
from fontTools.misc.roundTools import roundFunc
from fontTools.misc.psCharStrings import T2CharString, T2OutlineExtractor
from fontTools.pens.t2CharStringPen import T2CharStringPen
from functools import partial
from .errors import (
def merge_charstrings(glyphOrder, num_masters, top_dicts, masterModel):
    vsindex_dict = {}
    vsindex_by_key = {}
    varDataList = []
    masterSupports = []
    default_charstrings = top_dicts[0].CharStrings
    for gid, gname in enumerate(glyphOrder):
        all_cs = [_get_cs(td.CharStrings, gname, i != 0) for i, td in enumerate(top_dicts)]
        model, model_cs = masterModel.getSubModel(all_cs)
        default_charstring = model_cs[0]
        var_pen = CFF2CharStringMergePen([], gname, num_masters, 0)
        default_charstring.outlineExtractor = MergeOutlineExtractor
        default_charstring.draw(var_pen)
        region_cs = model_cs[1:]
        for region_idx, region_charstring in enumerate(region_cs, start=1):
            var_pen.restart(region_idx)
            region_charstring.outlineExtractor = MergeOutlineExtractor
            region_charstring.draw(var_pen)
        new_cs = var_pen.getCharString(private=default_charstring.private, globalSubrs=default_charstring.globalSubrs, var_model=model, optimize=True)
        default_charstrings[gname] = new_cs
        if not region_cs:
            continue
        if not var_pen.seen_moveto or 'blend' not in new_cs.program:
            continue
        key = tuple((v is not None for v in all_cs))
        try:
            vsindex = vsindex_by_key[key]
        except KeyError:
            vsindex = _add_new_vsindex(model, key, masterSupports, vsindex_dict, vsindex_by_key, varDataList)
        if vsindex != 0:
            new_cs.program[:0] = [vsindex, 'vsindex']
    if not vsindex_dict:
        key = (True,) * num_masters
        _add_new_vsindex(masterModel, key, masterSupports, vsindex_dict, vsindex_by_key, varDataList)
    cvData = CVarData(varDataList=varDataList, masterSupports=masterSupports, vsindex_dict=vsindex_dict)
    return cvData