from collections import OrderedDict
import copy
import os
from textwrap import fill, dedent
from dash.development.base_component import _explicitize_args
from dash.exceptions import NonExistentEventException
from ._all_keywords import python_keywords
from ._collect_nodes import collect_nodes, filter_base_nodes
from .base_component import Component
def array_of():
    inner = js_to_py_type(type_object['value'])
    if inner:
        return 'list of ' + (inner + 's' if inner.split(' ')[0] != 'dict' else inner.replace('dict', 'dicts', 1))
    return 'list'