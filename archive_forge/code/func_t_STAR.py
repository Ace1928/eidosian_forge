import codecs
import re
import ply.lex
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import pickle
from pyomo.common.deprecation import deprecated
from pyomo.core.base.component_namer import (
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.reference import Reference
@ply.lex.TOKEN('\\*{1,2}')
def t_STAR(t):
    if len(t.value) == 1:
        t.value = slice(None)
    else:
        t.value = Ellipsis
    return t