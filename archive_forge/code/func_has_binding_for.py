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
def has_binding_for(self, interface: type) -> bool:
    return interface in self._bindings