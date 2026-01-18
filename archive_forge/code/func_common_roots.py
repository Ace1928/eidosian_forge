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
def common_roots(paths):
    """
    Return a tuple of common roots that are shared between the given paths.
    File system watchers operate on directories and aren't cheap to create.
    Try to find the minimum set of directories to watch that encompass all of
    the files that need to be watched.
    """
    path_parts = sorted([x.parts for x in paths], key=len, reverse=True)
    tree = {}
    for chunks in path_parts:
        node = tree
        for chunk in chunks:
            node = node.setdefault(chunk, {})
        node.clear()

    def _walk(node, path):
        for prefix, child in node.items():
            yield from _walk(child, path + (prefix,))
        if not node:
            yield Path(*path)
    return tuple(_walk(tree, ()))