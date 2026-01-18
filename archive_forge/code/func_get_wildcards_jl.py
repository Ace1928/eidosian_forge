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
def get_wildcards_jl(props):
    return [key.replace('-*', '') for key in props if key.endswith('-*')]