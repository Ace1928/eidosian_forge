from __future__ import print_function
import argparse
import os
import sys
from importlib import import_module
from jinja2 import Template
from palettable.palette import Palette
def palette_name_sort_key(name):
    base, length = name.rsplit('_', maxsplit=1)
    return (base, int(length))