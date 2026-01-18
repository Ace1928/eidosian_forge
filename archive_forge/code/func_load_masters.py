from typing import List
from fontTools.misc.vector import Vector
from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.fixedTools import floatToFixed as fl2fi
from fontTools.misc.textTools import Tag, tostr
from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables._f_v_a_r import Axis, NamedInstance
from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates, dropImpliedOnCurvePoints
from fontTools.ttLib.tables.ttProgram import Program
from fontTools.ttLib.tables.TupleVariation import TupleVariation
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import OTTableWriter
from fontTools.varLib import builder, models, varStore
from fontTools.varLib.merger import VariationMerger, COLRVariationMerger
from fontTools.varLib.mvar import MVAR_ENTRIES
from fontTools.varLib.iup import iup_delta_optimize
from fontTools.varLib.featureVars import addFeatureVariations
from fontTools.designspaceLib import DesignSpaceDocument, InstanceDescriptor
from fontTools.designspaceLib.split import splitInterpolable, splitVariableFonts
from fontTools.varLib.stat import buildVFStatTable
from fontTools.colorLib.builder import buildColrV1
from fontTools.colorLib.unbuilder import unbuildColrV1
from functools import partial
from collections import OrderedDict, defaultdict, namedtuple
import os.path
import logging
from copy import deepcopy
from pprint import pformat
from re import fullmatch
from .errors import VarLibError, VarLibValidationError
def load_masters(designspace, master_finder=lambda s: s):
    """Ensure that all SourceDescriptor.font attributes have an appropriate TTFont
    object loaded, or else open TTFont objects from the SourceDescriptor.path
    attributes.

    The paths can point to either an OpenType font, a TTX file, or a UFO. In the
    latter case, use the provided master_finder callable to map from UFO paths to
    the respective master font binaries (e.g. .ttf, .otf or .ttx).

    Return list of master TTFont objects in the same order they are listed in the
    DesignSpaceDocument.
    """
    for master in designspace.sources:
        if master.layerName and master.font is None:
            raise VarLibValidationError(f"Designspace source '{master.name or '<Unknown>'}' specified a layer name but lacks the required TTFont object in the 'font' attribute.")
    return designspace.loadSourceFonts(_open_font, master_finder=master_finder)