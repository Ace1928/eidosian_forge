from collections import OrderedDict
import copy
import os
from textwrap import fill, dedent
from dash.development.base_component import _explicitize_args
from dash.exceptions import NonExistentEventException
from ._all_keywords import python_keywords
from ._collect_nodes import collect_nodes, filter_base_nodes
from .base_component import Component
def map_js_to_py_types_prop_types(type_object, indent_num):
    """Mapping from the PropTypes js type object to the Python type."""

    def shape_or_exact():
        return 'dict with keys:\n' + '\n'.join((create_prop_docstring(prop_name=prop_name, type_object=prop, required=prop['required'], description=prop.get('description', ''), default=prop.get('defaultValue'), indent_num=indent_num + 2) for prop_name, prop in sorted(list(type_object['value'].items()))))

    def array_of():
        inner = js_to_py_type(type_object['value'])
        if inner:
            return 'list of ' + (inner + 's' if inner.split(' ')[0] != 'dict' else inner.replace('dict', 'dicts', 1))
        return 'list'

    def tuple_of():
        elements = [js_to_py_type(element) for element in type_object['elements']]
        return f'list of {len(elements)} elements: [{', '.join(elements)}]'
    return dict(array=lambda: 'list', bool=lambda: 'boolean', number=lambda: 'number', string=lambda: 'string', object=lambda: 'dict', any=lambda: 'boolean | number | string | dict | list', element=lambda: 'dash component', node=lambda: 'a list of or a singular dash component, string or number', enum=lambda: 'a value equal to: ' + ', '.join((str(t['value']) for t in type_object['value'])), union=lambda: ' | '.join((js_to_py_type(subType) for subType in type_object['value'] if js_to_py_type(subType) != '')), arrayOf=array_of, objectOf=lambda: 'dict with strings as keys and values of type ' + js_to_py_type(type_object['value']), shape=shape_or_exact, exact=shape_or_exact, tuple=tuple_of)