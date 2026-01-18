import asyncio
import contextlib
import io
import inspect
import pprint
import sys
import builtins
import pkgutil
from asyncio import iscoroutinefunction
from types import CodeType, ModuleType, MethodType
from unittest.util import safe_repr
from functools import wraps, partial
from threading import RLock
@contextlib.contextmanager
def decoration_helper(self, patched, args, keywargs):
    extra_args = []
    with contextlib.ExitStack() as exit_stack:
        for patching in patched.patchings:
            arg = exit_stack.enter_context(patching)
            if patching.attribute_name is not None:
                keywargs.update(arg)
            elif patching.new is DEFAULT:
                extra_args.append(arg)
        args += tuple(extra_args)
        yield (args, keywargs)