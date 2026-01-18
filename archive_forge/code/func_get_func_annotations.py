from __future__ import annotations
from warnings import warn
import inspect
from .conflict import ordering, ambiguities, super_signature, AmbiguityWarning
from .utils import expand_tuples
import itertools as itl
@classmethod
def get_func_annotations(cls, func):
    """ Get annotations of function positional parameters
        """
    params = cls.get_func_params(func)
    if params:
        Parameter = inspect.Parameter
        params = (param for param in params if param.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD))
        annotations = tuple((param.annotation for param in params))
        if not any((ann is Parameter.empty for ann in annotations)):
            return annotations