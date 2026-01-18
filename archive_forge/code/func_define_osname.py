import cffi  # type: ignore
import os
import re
import sys
import warnings
import situation  # preloaded in setup.py
import importlib
def define_osname(definitions):
    if os.name == 'nt':
        definitions['OSNAME_NT'] = True