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
def component_data_objects(self, ctype=None, active=None, sort=False, descend_into=True, descent_order=None):
    """
        Return a generator that iterates through the
        component data objects for all components in a
        block.  By default, this generator recursively
        descends into sub-blocks.
        """
    dedup = _DeduplicateInfo()
    for _block in self.block_data_objects(active, sort, descend_into, descent_order):
        yield from _block._component_data_itervalues(ctype, active, sort, dedup)