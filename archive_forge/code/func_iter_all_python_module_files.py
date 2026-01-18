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
def iter_all_python_module_files():
    keys = sorted(sys.modules)
    modules = tuple((m for m in map(sys.modules.__getitem__, keys) if not isinstance(m, weakref.ProxyTypes)))
    return iter_modules_and_files(modules, frozenset(_error_files))