from collections import OrderedDict
import copy
import os
from textwrap import fill, dedent
from dash.development.base_component import _explicitize_args
from dash.exceptions import NonExistentEventException
from ._all_keywords import python_keywords
from ._collect_nodes import collect_nodes, filter_base_nodes
from .base_component import Component
def js_to_py_type(type_object, is_flow_type=False, indent_num=0):
    """Convert JS types to Python types for the component definition.
    Parameters
    ----------
    type_object: dict
        react-docgen-generated prop type dictionary
    is_flow_type: bool
        Does the prop use Flow types? Otherwise, uses PropTypes
    indent_num: int
        Number of indents to use for the docstring for the prop
    Returns
    -------
    str
        Python type string
    """
    js_type_name = type_object['name']
    js_to_py_types = map_js_to_py_types_flow_types(type_object=type_object) if is_flow_type else map_js_to_py_types_prop_types(type_object=type_object, indent_num=indent_num)
    if 'computed' in type_object and type_object['computed'] or type_object.get('type', '') == 'function':
        return ''
    if js_type_name in js_to_py_types:
        if js_type_name == 'signature':
            return js_to_py_types[js_type_name](indent_num)
        return js_to_py_types[js_type_name]()
    return ''