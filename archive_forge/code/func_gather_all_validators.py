import warnings
from collections import ChainMap
from functools import partial, partialmethod, wraps
from itertools import chain
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, Union, overload
from .errors import ConfigError
from .typing import AnyCallable
from .utils import ROOT_KEY, in_ipython
def gather_all_validators(type_: 'ModelOrDc') -> Dict[str, 'AnyClassMethod']:
    all_attributes = ChainMap(*[cls.__dict__ for cls in type_.__mro__])
    return {k: v for k, v in all_attributes.items() if hasattr(v, VALIDATOR_CONFIG_KEY) or hasattr(v, ROOT_VALIDATOR_CONFIG_KEY)}