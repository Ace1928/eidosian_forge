import inspect
import os
import re
import textwrap
import typing
from typing import Union
import warnings
from collections import OrderedDict
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
import rpy2.rinterface_lib.sexp
from rpy2.robjects import help
from rpy2.robjects import conversion
from rpy2.robjects.vectors import Vector
from rpy2.robjects.packages_utils import (default_symbol_r2python,
def map_signature(r_func: SignatureTranslatedFunction, is_method: bool=False, map_default: typing.Optional[typing.Callable[[rinterface.Sexp], typing.Any]]=_map_default_value) -> typing.Tuple[inspect.Signature, typing.Optional[int]]:
    """
    Map the signature of an function to the signature of a Python function.

    While mapping the signature, it will report the eventual presence of
    an R ellipsis.

    Args:
        r_func (SignatureTranslatedFunction): an R function
        is_method (bool): Whether the function should be treated as a method
            (adds a `self` param to the signature if so).
        map_default (function): Function to map default values in the Python
            signature. No mapping to default values is done if None.
    Returns:
        A tuple (inspect.Signature, int or None).
    """
    params = []
    r_ellipsis = None
    if is_method:
        params.append(inspect.Parameter('self', inspect.Parameter.POSITIONAL_ONLY))
    r_params = r_func.formals()
    rev_prm_transl = {v: k for k, v in r_func._prm_translate.items()}
    if r_params.names is not rinterface.NULL:
        for i, (name, default_orig) in enumerate(zip(r_params.names, r_params)):
            if default_orig == '...':
                r_ellipsis = i
                warnings.warn('The R ellispsis is not yet well supported.')
            transl_name = rev_prm_transl.get(name)
            if isinstance(default_orig, Vector):
                default_orig = default_orig[0]
                if map_default and (not rinterface.MissingArg.rsame(default_orig)):
                    default_mapped = map_default(default_orig)
                else:
                    default_mapped = inspect.Parameter.empty
            else:
                default_mapped = default_orig
            prm = inspect.Parameter(transl_name if transl_name else name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default_mapped)
            params.append(prm)
    return (inspect.Signature(params), r_ellipsis)