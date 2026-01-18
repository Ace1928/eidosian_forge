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
def mutable_param_validator(param_obj):
    """
    Check that Param-like object has attribute `mutable=True`.

    Parameters
    ----------
    param_obj : Param or _ParamData
        Param-like object of interest.

    Raises
    ------
    ValueError
        If lengths of the param object and the accompanying
        index set do not match. This may occur if some entry
        of the Param is not initialized.
    ValueError
        If attribute `mutable` is of value False.
    """
    if len(param_obj) != len(param_obj.index_set()):
        raise ValueError(f'Length of Param component object with name {param_obj.name!r} is {len(param_obj)}, and does not match that of its index set, which is of length {len(param_obj.index_set())}. Check that all entries of the component object have been initialized.')
    if not param_obj.mutable:
        raise ValueError(f'Param object with name {param_obj.name!r} is immutable.')