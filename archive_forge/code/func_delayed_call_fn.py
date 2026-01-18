import sys
from os import environ
from os.path import join
from copy import copy
from types import CodeType
from functools import partial
from kivy.factory import Factory
from kivy.lang.parser import (
from kivy.logger import Logger
from kivy.utils import QueryDict
from kivy.cache import Cache
from kivy import kivy_data_dir
from kivy.context import register_context
from kivy.resources import resource_find
from kivy._event import Observable, EventDispatcher
def delayed_call_fn(args, instance, v):
    if args[-1] is not None:
        return
    global _delayed_start
    if _delayed_start is None:
        _delayed_start = args
        args[-1] = StopIteration
    else:
        args[-1] = _delayed_start
        _delayed_start = args