from fontTools.misc.fixedTools import (
from fontTools.varLib.models import normalizeValue, piecewiseLinearMap
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables.TupleVariation import TupleVariation
from fontTools.ttLib.tables import _g_l_y_f
from fontTools import varLib
from fontTools import subset  # noqa: F401
from fontTools.varLib import builder
from fontTools.varLib.mvar import MVAR_ENTRIES
from fontTools.varLib.merger import MutatorMerger
from fontTools.varLib.instancer import names
from .featureVars import instantiateFeatureVariations
from fontTools.misc.cliTools import makeOutputFileName
from fontTools.varLib.instancer import solver
import collections
import dataclasses
from contextlib import contextmanager
from copy import deepcopy
from enum import IntEnum
import logging
import os
import re
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union
import warnings
def instantiateOTL(varfont, axisLimits):
    if 'GDEF' not in varfont or varfont['GDEF'].table.Version < 65539 or (not varfont['GDEF'].table.VarStore):
        return
    if 'GPOS' in varfont:
        msg = 'Instantiating GDEF and GPOS tables'
    else:
        msg = 'Instantiating GDEF table'
    log.info(msg)
    gdef = varfont['GDEF'].table
    varStore = gdef.VarStore
    fvarAxes = varfont['fvar'].axes
    defaultDeltas = instantiateItemVariationStore(varStore, fvarAxes, axisLimits)
    merger = MutatorMerger(varfont, defaultDeltas, deleteVariations=not varStore.VarRegionList.Region)
    merger.mergeTables(varfont, [varfont], ['GDEF', 'GPOS'])
    if varStore.VarRegionList.Region:
        varIndexMapping = varStore.optimize()
        gdef.remap_device_varidxes(varIndexMapping)
        if 'GPOS' in varfont:
            varfont['GPOS'].table.remap_device_varidxes(varIndexMapping)
    else:
        del gdef.VarStore
        gdef.Version = 65538
        if gdef.MarkGlyphSetsDef is None:
            del gdef.MarkGlyphSetsDef
            gdef.Version = 65536
        if not (gdef.LigCaretList or gdef.MarkAttachClassDef or gdef.GlyphClassDef or gdef.AttachList or (gdef.Version >= 65538 and gdef.MarkGlyphSetsDef)):
            del varfont['GDEF']