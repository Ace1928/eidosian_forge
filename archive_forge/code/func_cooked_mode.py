from __future__ import unicode_literals
from .utils import DummyContext, is_windows
from abc import ABCMeta, abstractmethod
from six import with_metaclass
import io
import os
import sys
def cooked_mode(self):
    return DummyContext()