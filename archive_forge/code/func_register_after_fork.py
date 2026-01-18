import os
import itertools
import sys
import weakref
import atexit
import threading        # we want threading to install it's
from subprocess import _args_from_interpreter_flags
from . import process
def register_after_fork(obj, func):
    _afterfork_registry[next(_afterfork_counter), id(obj), func] = obj