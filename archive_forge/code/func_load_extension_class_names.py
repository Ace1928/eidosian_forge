import os
import sys
import warnings
from contextlib import contextmanager
from importlib import import_module, reload
from kombu.utils.imports import symbol_by_name
def load_extension_class_names(namespace):
    if sys.version_info >= (3, 10):
        _entry_points = entry_points(group=namespace)
    else:
        try:
            _entry_points = entry_points().get(namespace, [])
        except AttributeError:
            _entry_points = entry_points().select(group=namespace)
    for ep in _entry_points:
        yield (ep.name, ep.value)