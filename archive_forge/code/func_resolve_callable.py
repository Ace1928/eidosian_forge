import inspect
import itertools
import logging
import warnings
from collections import OrderedDict, defaultdict, deque
from functools import partial
from six import string_types
@staticmethod
def resolve_callable(func, event_data):
    """ Converts a model's property name, method name or a path to a callable into a callable.
            If func is not a string it will be returned unaltered.
        Args:
            func (str or callable): Property name, method name or a path to a callable
            event_data (EventData): Currently processed event
        Returns:
            callable function resolved from string or func
        """
    if isinstance(func, string_types):
        try:
            func = getattr(event_data.model, func)
            if not callable(func):

                def func_wrapper(*_, **__):
                    return func
                return func_wrapper
        except AttributeError:
            try:
                module_name, func_name = func.rsplit('.', 1)
                module = __import__(module_name)
                for submodule_name in module_name.split('.')[1:]:
                    module = getattr(module, submodule_name)
                func = getattr(module, func_name)
            except (ImportError, AttributeError, ValueError):
                raise AttributeError("Callable with name '%s' could neither be retrieved from the passed model nor imported from a module." % func)
    return func