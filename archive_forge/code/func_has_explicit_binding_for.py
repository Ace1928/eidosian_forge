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
def has_explicit_binding_for(self, interface: type) -> bool:
    return self.has_binding_for(interface) and (not isinstance(self._bindings[interface], ImplicitBinding))