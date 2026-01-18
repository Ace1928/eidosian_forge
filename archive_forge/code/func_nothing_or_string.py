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
def nothing_or_string(v):
    return '"{}"'.format(v) if v else 'nothing'