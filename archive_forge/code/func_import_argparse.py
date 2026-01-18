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
def import_argparse(self, parsed_args):
    for level, prefix, value, obj in self._data_collector(None, ''):
        if obj._argparse is None:
            continue
        for _args, _kwds in obj._argparse:
            if 'dest' in _kwds:
                _dest = _kwds['dest']
                if _dest in parsed_args:
                    obj.set_value(parsed_args.__dict__[_dest])
            else:
                _dest = 'CONFIGBLOCK.' + obj.name(True)
                if _dest in parsed_args:
                    obj.set_value(parsed_args.__dict__[_dest])
                    del parsed_args.__dict__[_dest]
    return parsed_args