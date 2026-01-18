from fontTools.misc.fixedTools import floatToFixedToFloat, floatToFixed
from fontTools.misc.roundTools import otRound
from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables import ttProgram
from fontTools.ttLib.tables._g_l_y_f import (
from fontTools.varLib.models import (
from fontTools.varLib.merger import MutatorMerger
from fontTools.varLib.varStore import VarStoreInstancer
from fontTools.varLib.mvar import MVAR_ENTRIES
from fontTools.varLib.iup import iup_delta
import fontTools.subset.cff
import os.path
import logging
from io import BytesIO
def interpolate_cff2_PrivateDict(topDict, interpolateFromDeltas):
    pd_blend_lists = ('BlueValues', 'OtherBlues', 'FamilyBlues', 'FamilyOtherBlues', 'StemSnapH', 'StemSnapV')
    pd_blend_values = ('BlueScale', 'BlueShift', 'BlueFuzz', 'StdHW', 'StdVW')
    for fontDict in topDict.FDArray:
        pd = fontDict.Private
        vsindex = pd.vsindex if hasattr(pd, 'vsindex') else 0
        for key, value in pd.rawDict.items():
            if key in pd_blend_values and isinstance(value, list):
                delta = interpolateFromDeltas(vsindex, value[1:])
                pd.rawDict[key] = otRound(value[0] + delta)
            elif key in pd_blend_lists and isinstance(value[0], list):
                'If any argument in a BlueValues list is a blend list,\n                then they all are. The first value of each list is an\n                absolute value. The delta tuples are calculated from\n                relative master values, hence we need to append all the\n                deltas to date to each successive absolute value.'
                delta = 0
                for i, val_list in enumerate(value):
                    delta += otRound(interpolateFromDeltas(vsindex, val_list[1:]))
                    value[i] = val_list[0] + delta