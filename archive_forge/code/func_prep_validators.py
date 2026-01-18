import warnings
from collections import ChainMap
from functools import partial, partialmethod, wraps
from itertools import chain
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, Union, overload
from .errors import ConfigError
from .typing import AnyCallable
from .utils import ROOT_KEY, in_ipython
def prep_validators(v_funcs: Iterable[AnyCallable]) -> 'ValidatorsList':
    return [make_generic_validator(f) for f in v_funcs if f]