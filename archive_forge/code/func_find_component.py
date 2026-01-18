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
def find_component(self, label_or_component):
    """
        Returns a component in the block given a name.

        Parameters
        ----------
        label_or_component : str, Component, or ComponentUID
            The name of the component to find in this block. String or
            Component arguments are first converted to ComponentUID.

        Returns
        -------
        Component
            Component on the block identified by the ComponentUID. If
            a matching component is not found, None is returned.

        """
    if type(label_or_component) is ComponentUID:
        cuid = label_or_component
    else:
        cuid = ComponentUID(label_or_component)
    return cuid.find_component_on(self)