from asyncio import iscoroutinefunction
from contextlib import contextmanager
from functools import partial, wraps
from types import coroutine
import builtins
import inspect
import linecache
import logging
import os
import io
import pdb
import subprocess
import sys
import time
import traceback
import warnings
import psutil
@classmethod
def register_magics(cls, ip):
    from distutils.version import LooseVersion
    import IPython
    ipython_version = LooseVersion(IPython.__version__)
    if ipython_version < '0.13':
        try:
            _register_magic = ip.define_magic
        except AttributeError:
            _register_magic = ip.expose_magic
        _register_magic('mprun', cls.mprun.__func__)
        _register_magic('memit', cls.memit.__func__)
    else:
        ip.register_magics(cls)