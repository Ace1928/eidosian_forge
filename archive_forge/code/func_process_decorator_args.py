from __future__ import annotations
from functools import wraps
import zmq
def process_decorator_args(self, *args, **kwargs):
    """Also grab context_name out of kwargs"""
    kw_name, args, kwargs = super().process_decorator_args(*args, **kwargs)
    self.context_name = kwargs.pop('context_name', 'context')
    return (kw_name, args, kwargs)