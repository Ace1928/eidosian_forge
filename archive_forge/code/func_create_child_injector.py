import functools
import inspect
import itertools
import logging
import sys
import threading
import types
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import (
def create_child_injector(self, *args: Any, **kwargs: Any) -> 'Injector':
    kwargs['parent'] = self
    return Injector(*args, **kwargs)