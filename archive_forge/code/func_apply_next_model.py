import functools
import sys
import threading
import warnings
from collections import Counter, defaultdict
from functools import partial
from django.core.exceptions import AppRegistryNotReady, ImproperlyConfigured
from .config import AppConfig
def apply_next_model(model):
    next_function = partial(apply_next_model.func, model)
    self.lazy_model_operation(next_function, *more_models)