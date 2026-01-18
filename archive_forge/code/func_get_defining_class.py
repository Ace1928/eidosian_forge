import argparse
import collections
import functools
import glob
import inspect
import itertools
import os
import re
import subprocess
import sys
import threading
import unicodedata
from enum import (
from typing import (
from . import (
from .argparse_custom import (
def get_defining_class(meth: Callable[..., Any]) -> Optional[Type[Any]]:
    """
    Attempts to resolve the class that defined a method.

    Inspired by implementation published here:
        https://stackoverflow.com/a/25959545/1956611

    :param meth: method to inspect
    :return: class type in which the supplied method was defined. None if it couldn't be resolved.
    """
    if isinstance(meth, functools.partial):
        return get_defining_class(meth.func)
    if inspect.ismethod(meth) or (inspect.isbuiltin(meth) and getattr(meth, '__self__') is not None and getattr(meth.__self__, '__class__')):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
        meth = getattr(meth, '__func__', meth)
    if inspect.isfunction(meth):
        cls = getattr(inspect.getmodule(meth), meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0])
        if isinstance(cls, type):
            return cls
    return cast(type, getattr(meth, '__objclass__', None))