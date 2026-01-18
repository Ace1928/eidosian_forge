from __future__ import annotations
from . import coredata as cdata
from .mesonlib import MachineChoice, OptionKey
import os.path
import pprint
import textwrap
def print_dep(dep_key, dep):
    print('  ' + dep_key[0][1] + ': ')
    print('      compile args: ' + repr(dep.get_compile_args()))
    print('      link args: ' + repr(dep.get_link_args()))
    if dep.get_sources():
        print('      sources: ' + repr(dep.get_sources()))
    print('      version: ' + repr(dep.get_version()))