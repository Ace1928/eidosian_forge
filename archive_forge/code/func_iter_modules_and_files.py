import itertools
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
import weakref
from collections import defaultdict
from functools import lru_cache, wraps
from pathlib import Path
from types import ModuleType
from zipimport import zipimporter
import django
from django.apps import apps
from django.core.signals import request_finished
from django.dispatch import Signal
from django.utils.functional import cached_property
from django.utils.version import get_version_tuple
@lru_cache(maxsize=1)
def iter_modules_and_files(modules, extra_files):
    """Iterate through all modules needed to be watched."""
    sys_file_paths = []
    for module in modules:
        if not isinstance(module, ModuleType):
            continue
        if module.__name__ in ('__main__', '__mp_main__'):
            if hasattr(module, '__file__'):
                sys_file_paths.append(module.__file__)
            continue
        if getattr(module, '__spec__', None) is None:
            continue
        spec = module.__spec__
        if spec.has_location:
            origin = spec.loader.archive if isinstance(spec.loader, zipimporter) else spec.origin
            sys_file_paths.append(origin)
    results = set()
    for filename in itertools.chain(sys_file_paths, extra_files):
        if not filename:
            continue
        path = Path(filename)
        try:
            if not path.exists():
                continue
        except ValueError as e:
            logger.debug('"%s" raised when resolving path: "%s"', e, path)
            continue
        resolved_path = path.resolve().absolute()
        results.add(resolved_path)
    return frozenset(results)