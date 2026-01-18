import codecs
import re
import ply.lex
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import pickle
from pyomo.common.deprecation import deprecated
from pyomo.core.base.component_namer import (
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.reference import Reference
@ply.lex.TOKEN('\\|b?(?:' + _general_str + ')')
def t_PICKLE(t):
    start = 3 if t.value[1] == 'b' else 2
    unescaped = _re_escape_sequences.sub(_match_escape, t.value[start:-1])
    rawstr = bytes(list((ord(_) for _ in unescaped)))
    t.value = pickle.loads(rawstr)
    return t