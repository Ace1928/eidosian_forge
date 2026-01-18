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
def setRibbiBits(font):
    """Set the `head.macStyle` and `OS/2.fsSelection` style bits
    appropriately."""
    english_ribbi_style = font['name'].getName(names.NameID.SUBFAMILY_NAME, 3, 1, 1033)
    if english_ribbi_style is None:
        return
    styleMapStyleName = english_ribbi_style.toStr().lower()
    if styleMapStyleName not in {'regular', 'bold', 'italic', 'bold italic'}:
        return
    if styleMapStyleName == 'bold':
        font['head'].macStyle = 1
    elif styleMapStyleName == 'bold italic':
        font['head'].macStyle = 3
    elif styleMapStyleName == 'italic':
        font['head'].macStyle = 2
    selection = font['OS/2'].fsSelection
    selection &= ~(1 << 0)
    selection &= ~(1 << 5)
    selection &= ~(1 << 6)
    if styleMapStyleName == 'regular':
        selection |= 1 << 6
    elif styleMapStyleName == 'bold':
        selection |= 1 << 5
    elif styleMapStyleName == 'italic':
        selection |= 1 << 0
    elif styleMapStyleName == 'bold italic':
        selection |= 1 << 0
        selection |= 1 << 5
    font['OS/2'].fsSelection = selection