from __future__ import (absolute_import, division, print_function)
import importlib
import os
import re
import sys
import textwrap
import yaml
def doc_fragment_fn(name):
    return os.path.join('plugins', 'doc_fragments', '{0}.py'.format(name))