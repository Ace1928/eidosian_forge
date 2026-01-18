import os
import sys
import shutil
import importlib
import textwrap
import re
import warnings
from ._all_keywords import r_keywords
from ._py_components_generation import reorder_props
def shape_or_exact():
    return 'lists containing elements {}.\n{}'.format(', '.join(("'{}'".format(t) for t in type_object['value'])), 'Those elements have the following types:\n{}'.format('\n'.join((create_prop_docstring_r(prop_name=prop_name, type_object=prop, required=prop['required'], description=prop.get('description', ''), indent_num=1) for prop_name, prop in type_object['value'].items()))))