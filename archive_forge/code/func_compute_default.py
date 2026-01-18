import collections
import copy
import datetime as dt
import glob
import inspect
import numbers
import os.path
import pathlib
import re
import sys
import typing
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from .parameterized import (
from ._utils import (
def compute_default(self):
    if self.default is None and callable(self.compute_default_fn):
        self.default = self.compute_default_fn()
        for o in self.default:
            if o not in self.objects:
                self.objects.append(o)