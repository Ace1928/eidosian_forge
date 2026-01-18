import functools
import warnings
import json
import contextvars
import flask
from . import exceptions
from ._utils import AttributeDict
@property
@has_context
def outputs_list(self):
    if self.using_outputs_grouping:
        warnings.warn('outputs_list is deprecated, use outputs_grouping instead', DeprecationWarning)
    return getattr(_get_context_value(), 'outputs_list', [])