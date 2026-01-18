import logging
import sys
from copy import deepcopy
from pickle import PickleError
from weakref import ref as weakref_ref
import pyomo.common
from pyomo.common import DeveloperError
from pyomo.common.autoslots import AutoSlots, fast_deepcopy
from pyomo.common.collections import OrderedDict
from pyomo.common.deprecation import (
from pyomo.common.factory import Factory
from pyomo.common.formatting import tabular_writer, StreamIndenter
from pyomo.common.modeling import NOTSET
from pyomo.common.sorting import sorted_robust
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.base.component_namer import name_repr, index_repr
from pyomo.core.base.global_set import UnindexedComponent_index
@deprecated('The cname() method has been renamed to getname().\n    The preferred method of obtaining a component name is to use the\n    .name property, which returns the fully qualified component name.\n    The .local_name property will return the component name only within\n    the context of the immediate parent container.', version='5.0')
def cname(self, *args, **kwds):
    return self.getname(*args, **kwds)