import logging
import os
from collections import defaultdict, namedtuple
from functools import reduce
from itertools import chain
from math import log2
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple
from fontTools.config import OPTIONS
from fontTools.misc.intTools import bit_count, bit_indices
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables import otBase, otTables
def is_really_zero(class2: otTables.Class2Record) -> bool:
    v1 = getattr(class2, 'Value1', None)
    v2 = getattr(class2, 'Value2', None)
    return (v1 is None or v1.getEffectiveFormat() == 0) and (v2 is None or v2.getEffectiveFormat() == 0)