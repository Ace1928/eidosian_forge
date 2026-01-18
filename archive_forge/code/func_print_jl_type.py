import copy
import os
import shutil
import warnings
import sys
import importlib
import uuid
import hashlib
from ._all_keywords import julia_keywords
from ._py_components_generation import reorder_props
def print_jl_type(typedata):
    typestring = get_jl_type(typedata).capitalize()
    if typestring:
        typestring += '. '
    return typestring