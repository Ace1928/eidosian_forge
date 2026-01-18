from __future__ import print_function
import sys
import os
import types
import traceback
from abc import abstractmethod
def set_max_print(self, i):
    self._max_print = i
    self._count = None