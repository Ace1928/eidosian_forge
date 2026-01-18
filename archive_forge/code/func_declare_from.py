import argparse
import builtins
import enum
import importlib
import inspect
import io
import logging
import os
import pickle
import ply.lex
import re
import sys
import textwrap
import types
from operator import attrgetter
from pyomo.common.collections import Sequence, Mapping
from pyomo.common.deprecation import (
from pyomo.common.fileutils import import_file
from pyomo.common.formatting import wrap_reStructuredText
from pyomo.common.modeling import NOTSET
def declare_from(self, other, skip=None):
    if not isinstance(other, ConfigDict):
        raise ValueError('ConfigDict.declare_from() only accepts other ConfigDicts')
    for key in other.keys():
        if skip and key in skip:
            continue
        if key in self:
            raise ValueError("ConfigDict.declare_from passed a block with a duplicate field, '%s'" % (key,))
        self.declare(key, other.get(key)())