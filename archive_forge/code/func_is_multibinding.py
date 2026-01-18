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
def is_multibinding(self) -> bool:
    return _get_origin(_punch_through_alias(self.interface)) in {dict, list}