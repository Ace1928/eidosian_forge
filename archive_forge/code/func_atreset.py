import logging
import types
import weakref
from pyomo.common.pyomo_typing import overload
from ctypes import (
from pyomo.common.autoslots import AutoSlots
from pyomo.common.fileutils import find_library
from pyomo.core.expr.numvalue import (
import pyomo.core.expr as EXPR
from pyomo.core.base.component import Component
from pyomo.core.base.units_container import units
def atreset(ae, a, b):
    logger.warning('AMPL External function: ignoring AtReset call in external library.  This may result in a memory leak or other undesirable behavior.')