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
def standardize_cdatatype_obj(self, obj):
    """
        Standardize object of type ``self.cdatatype`` to
        ``[obj]``.
        """
    if self.cdatatype_validator is not None:
        self.cdatatype_validator(obj)
    return [obj]