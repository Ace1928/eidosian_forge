import warnings
from collections import ChainMap
from functools import partial, partialmethod, wraps
from itertools import chain
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, Union, overload
from .errors import ConfigError
from .typing import AnyCallable
from .utils import ROOT_KEY, in_ipython
def make_generic_validator(validator: AnyCallable) -> 'ValidatorCallable':
    """
    Make a generic function which calls a validator with the right arguments.

    Unfortunately other approaches (eg. return a partial of a function that builds the arguments) is slow,
    hence this laborious way of doing things.

    It's done like this so validators don't all need **kwargs in their signature, eg. any combination of
    the arguments "values", "fields" and/or "config" are permitted.
    """
    from inspect import signature
    if not isinstance(validator, (partial, partialmethod)):
        sig = signature(validator)
        args = list(sig.parameters.keys())
    else:
        sig = signature(validator.func)
        args = [k for k in signature(validator.func).parameters.keys() if k not in validator.args | validator.keywords.keys()]
    first_arg = args.pop(0)
    if first_arg == 'self':
        raise ConfigError(f'Invalid signature for validator {validator}: {sig}, "self" not permitted as first argument, should be: (cls, value, values, config, field), "values", "config" and "field" are all optional.')
    elif first_arg == 'cls':
        return wraps(validator)(_generic_validator_cls(validator, sig, set(args[1:])))
    else:
        return wraps(validator)(_generic_validator_basic(validator, sig, set(args)))