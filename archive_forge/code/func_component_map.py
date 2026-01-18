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
def component_map(self, ctype=None, active=None, sort=False):
    """Returns a PseudoMap of the components in this block.

        Parameters
        ----------
        ctype:  None or type or iterable
            Specifies the component types (`ctypes`) to include in the
            resulting PseudoMap

                =============   ===================================
                None            All components
                type            A single component type
                iterable        All component types in the iterable
                =============   ===================================

        active: None or bool
            Filter components by the active flag

                =====  ===============================
                None   Return all components
                True   Return only active components
                False  Return only inactive components
                =====  ===============================

        sort: bool
            Iterate over the components in a sorted order

                =====  ================================================
                True   Iterate using Block.alphabetizeComponentAndIndex
                False  Iterate using Block.declarationOrder
                =====  ================================================

        """
    return PseudoMap(self, ctype, active, sort)