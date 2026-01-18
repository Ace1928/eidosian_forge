import functools
import sys
import threading
import warnings
from collections import Counter, defaultdict
from functools import partial
from django.core.exceptions import AppRegistryNotReady, ImproperlyConfigured
from .config import AppConfig
def lazy_model_operation(self, function, *model_keys):
    """
        Take a function and a number of ("app_label", "modelname") tuples, and
        when all the corresponding models have been imported and registered,
        call the function with the model classes as its arguments.

        The function passed to this method must accept exactly n models as
        arguments, where n=len(model_keys).
        """
    if not model_keys:
        function()
    else:
        next_model, *more_models = model_keys

        def apply_next_model(model):
            next_function = partial(apply_next_model.func, model)
            self.lazy_model_operation(next_function, *more_models)
        apply_next_model.func = function
        try:
            model_class = self.get_registered_model(*next_model)
        except LookupError:
            self._pending_operations[next_model].append(apply_next_model)
        else:
            apply_next_model(model_class)