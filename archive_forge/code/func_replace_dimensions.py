import copy
import sys
from functools import wraps
from types import FunctionType
import param
from . import util
from .pprint import PrettyPrinter
@classmethod
def replace_dimensions(cls, dimensions, overrides):
    """Replaces dimensions in list with dictionary of overrides.

        Args:
            dimensions: List of dimensions
            overrides: Dictionary of dimension specs indexed by name

        Returns:
            list: List of dimensions with replacements applied
        """
    from .dimension import Dimension
    replaced = []
    for d in dimensions:
        if d.name in overrides:
            override = overrides[d.name]
        elif d.label in overrides:
            override = overrides[d.label]
        else:
            override = None
        if override is None:
            replaced.append(d)
        elif isinstance(override, (str, tuple)):
            replaced.append(d.clone(override))
        elif isinstance(override, Dimension):
            replaced.append(override)
        elif isinstance(override, dict):
            replaced.append(d.clone(override.get('name', None), **{k: v for k, v in override.items() if k != 'name'}))
        else:
            raise ValueError('Dimension can only be overridden with another dimension or a dictionary of attributes')
    return replaced