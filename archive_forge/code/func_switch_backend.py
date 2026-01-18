import os
import re
import sys
import ctypes
import textwrap
from typing import final
import warnings
from ctypes.util import find_library
from abc import ABC, abstractmethod
from functools import lru_cache
from contextlib import ContextDecorator
def switch_backend(self, backend):
    """Switch the backend of FlexiBLAS

        Parameters
        ----------
        backend : str
            The name or the path to the shared library of the backend to switch to. If
            the backend is not already loaded, it will be loaded first.
        """
    if backend not in self.loaded_backends:
        if backend in self.available_backends:
            load_func = getattr(self.dynlib, 'flexiblas_load_backend', lambda _: -1)
        else:
            load_func = getattr(self.dynlib, 'flexiblas_load_backend_library', lambda _: -1)
        res = load_func(str(backend).encode('utf-8'))
        if res == -1:
            raise RuntimeError(f'Failed to load backend {backend!r}. It must either be the name of a backend available in the FlexiBLAS configuration {self.available_backends} or the path to a valid shared library.')
        self.parent._load_libraries()
    switch_func = getattr(self.dynlib, 'flexiblas_switch', lambda _: -1)
    idx = self.loaded_backends.index(backend)
    res = switch_func(idx)
    if res == -1:
        raise RuntimeError(f'Failed to switch to backend {backend!r}.')