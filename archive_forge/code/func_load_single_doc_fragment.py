from __future__ import (absolute_import, division, print_function)
import importlib
import os
import re
import sys
import textwrap
import yaml
def load_single_doc_fragment(name):
    fragment = 'DOCUMENTATION'
    if '.' in name:
        name, fragment = name.split('.', 1)
        fragment = fragment.upper()
    doc_fragment = load_doc_fragment(name)
    return doc_fragment.fragments_by_name[fragment]