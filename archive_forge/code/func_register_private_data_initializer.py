import copy
import logging
import sys
import weakref
import textwrap
from collections import defaultdict
from contextlib import contextmanager
from inspect import isclass, currentframe
from io import StringIO
from itertools import filterfalse, chain
from operator import itemgetter, attrgetter
from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import Mapping
from pyomo.common.deprecation import deprecated, deprecation_warning, RenamedClass
from pyomo.common.formatting import StreamIndenter
from pyomo.common.gc_manager import PauseGC
from pyomo.common.log import is_debug_set
from pyomo.common.pyomo_typing import overload
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.component import (
from pyomo.core.base.enums import SortComponents, TraversalStrategy
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.set import Any
from pyomo.core.base.var import Var
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.indexed_component import (
from pyomo.opt.base import ProblemFormat, guess_format
from pyomo.opt import WriterFactory
@staticmethod
def register_private_data_initializer(initializer, scope=None):
    mod = currentframe().f_back.f_globals['__name__']
    if scope is None:
        scope = mod
    elif not mod.startswith(scope):
        raise ValueError(f"'private_data' scope must be substrings of the caller's module name. Received '{scope}' when calling register_private_data_initializer().")
    if scope in Block._private_data_initializers:
        raise RuntimeError(f"Duplicate initializer registration for 'private_data' dictionary (scope={scope})")
    Block._private_data_initializers[scope] = initializer