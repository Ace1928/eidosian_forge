import warnings
import hashlib
import io
import json
import jsonschema
import pandas as pd
from toolz.curried import pipe as _pipe
import itertools
import sys
from typing import cast, List, Optional, Any, Iterable, Union, Literal, IO
from typing import Type as TypingType
from typing import Dict as TypingDict
from .schema import core, channels, mixins, Undefined, UndefinedType, SCHEMA_URL
from .data import data_transformers
from ... import utils, expr
from ...expr import core as _expr_core
from .display import renderers, VEGALITE_VERSION, VEGAEMBED_VERSION, VEGA_VERSION
from .theme import themes
from .compiler import vegalite_compilers
from ...utils._vegafusion_data import (
from ...utils.core import DataFrameLike
from ...utils.data import DataType
def param(name: Optional[str]=None, value: Union[Any, UndefinedType]=Undefined, bind: Union[core.Binding, UndefinedType]=Undefined, empty: Union[bool, UndefinedType]=Undefined, expr: Union[str, core.Expr, _expr_core.Expression, UndefinedType]=Undefined, **kwds) -> Parameter:
    """Create a named parameter.
    See https://altair-viz.github.io/user_guide/interactions.html for examples.
    Although both variable parameters and selection parameters can be created using
    this 'param' function, to create a selection parameter, it is recommended to use
    either 'selection_point' or 'selection_interval' instead.

    Parameters
    ----------
    name : string (optional)
        The name of the parameter. If not specified, a unique name will be
        created.
    value : any (optional)
        The default value of the parameter. If not specified, the parameter
        will be created without a default value.
    bind : :class:`Binding` (optional)
        Binds the parameter to an external input element such as a slider,
        selection list or radio button group.
    empty : boolean (optional)
        For selection parameters, the predicate of empty selections returns
        True by default. Override this behavior, by setting this property
        'empty=False'.
    expr : str, Expression (optional)
        An expression for the value of the parameter. This expression may
        include other parameters, in which case the parameter will
        automatically update in response to upstream parameter changes.
    **kwds :
        additional keywords will be used to construct a parameter.  If 'select'
        is among the keywords, then a selection parameter will be created.
        Otherwise, a variable parameter will be created.

    Returns
    -------
    parameter: Parameter
        The parameter object that can be used in chart creation.
    """
    parameter = Parameter(name)
    if empty is not Undefined:
        parameter.empty = empty
        if parameter.empty == 'none':
            warnings.warn("The value of 'empty' should be True or False.", utils.AltairDeprecationWarning, stacklevel=1)
            parameter.empty = False
        elif parameter.empty == 'all':
            warnings.warn("The value of 'empty' should be True or False.", utils.AltairDeprecationWarning, stacklevel=1)
            parameter.empty = True
        elif parameter.empty is False or parameter.empty is True:
            pass
        else:
            raise ValueError("The value of 'empty' should be True or False.")
    if 'init' in kwds:
        warnings.warn("Use 'value' instead of 'init'.", utils.AltairDeprecationWarning, stacklevel=1)
        if value is Undefined:
            kwds['value'] = kwds.pop('init')
        else:
            kwds.pop('init')
    if 'select' not in kwds:
        parameter.param = core.VariableParameter(name=parameter.name, bind=bind, value=value, expr=expr, **kwds)
        parameter.param_type = 'variable'
    elif 'views' in kwds:
        parameter.param = core.TopLevelSelectionParameter(name=parameter.name, bind=bind, value=value, expr=expr, **kwds)
        parameter.param_type = 'selection'
    else:
        parameter.param = core.SelectionParameter(name=parameter.name, bind=bind, value=value, expr=expr, **kwds)
        parameter.param_type = 'selection'
    return parameter