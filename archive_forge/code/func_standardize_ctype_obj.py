from collections.abc import Iterable
import logging
from pyomo.common.collections import ComponentSet
from pyomo.common.config import (
from pyomo.common.errors import ApplicationError, PyomoException
from pyomo.core.base import Var, _VarData
from pyomo.core.base.param import Param, _ParamData
from pyomo.opt import SolverFactory
from pyomo.contrib.pyros.util import ObjectiveType, setup_pyros_logger
from pyomo.contrib.pyros.uncertainty_sets import UncertaintySet
def standardize_ctype_obj(self, obj):
    """
        Standardize object of type ``self.ctype`` to list
        of objects of type ``self.cdatatype``.
        """
    if self.ctype_validator is not None:
        self.ctype_validator(obj)
    return list(obj.values())