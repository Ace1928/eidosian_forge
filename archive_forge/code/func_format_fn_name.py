import os
import sys
import shutil
import importlib
import textwrap
import re
import warnings
from ._all_keywords import r_keywords
from ._py_components_generation import reorder_props
def format_fn_name(prefix, name):
    if prefix:
        return prefix + snake_case_to_camel_case(name)
    return snake_case_to_camel_case(name[0].lower() + name[1:])