from __future__ import annotations
from . import coredata as cdata
from .mesonlib import MachineChoice, OptionKey
import os.path
import pprint
import textwrap
def dump_guids(d):
    for name, value in d.items():
        print('  ' + name + ': ' + value)