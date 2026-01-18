import collections
import copy
import enum
from functools import partial
from math import ceil, log
from typing import (
from fontTools.misc.arrayTools import intRect
from fontTools.misc.fixedTools import fixedToFloat
from fontTools.misc.treeTools import build_n_ary_tree
from fontTools.ttLib.tables import C_O_L_R_
from fontTools.ttLib.tables import C_P_A_L_
from fontTools.ttLib.tables import _n_a_m_e
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otTables import ExtendMode, CompositeMode
from .errors import ColorLibError
from .geometry import round_start_circle_stable_containment
from .table_builder import BuildCallback, TableBuilder
def buildClipList(clipBoxes: Dict[str, _ClipBoxInput]) -> ot.ClipList:
    clipList = ot.ClipList()
    clipList.Format = 1
    clipList.clips = {name: buildClipBox(box) for name, box in clipBoxes.items()}
    return clipList