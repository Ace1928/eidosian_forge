from pyomo.network.port import Port
from pyomo.core.base.component import ActiveComponentData, ModelComponentFactory
from pyomo.core.base.indexed_component import (
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.common.deprecation import RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.timing import ConstructionTimer
from weakref import ref as weakref_ref
import logging, sys
@property
def expanded_block(self):
    return self._expanded_block