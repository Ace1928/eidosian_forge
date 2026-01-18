from functools import wraps
from importlib import import_module
from inspect import getfullargspec, unwrap
from django.utils.html import conditional_escape
from django.utils.itercompat import is_iterable
from .base import Node, Template, token_kwargs
from .exceptions import TemplateSyntaxError
def get_resolved_arguments(self, context):
    resolved_args = [var.resolve(context) for var in self.args]
    if self.takes_context:
        resolved_args = [context] + resolved_args
    resolved_kwargs = {k: v.resolve(context) for k, v in self.kwargs.items()}
    return (resolved_args, resolved_kwargs)