import functools
import warnings
import json
import contextvars
import flask
from . import exceptions
from ._utils import AttributeDict
@property
@has_context
def states_list(self):
    if self.using_args_grouping:
        warnings.warn('states_list is deprecated, use args_grouping instead', DeprecationWarning)
    return getattr(_get_context_value(), 'states_list', [])