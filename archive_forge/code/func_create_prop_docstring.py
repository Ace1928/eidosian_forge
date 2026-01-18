from collections import OrderedDict
import copy
import os
from textwrap import fill, dedent
from dash.development.base_component import _explicitize_args
from dash.exceptions import NonExistentEventException
from ._all_keywords import python_keywords
from ._collect_nodes import collect_nodes, filter_base_nodes
from .base_component import Component
def create_prop_docstring(prop_name, type_object, required, description, default, indent_num, is_flow_type=False):
    """Create the Dash component prop docstring.
    Parameters
    ----------
    prop_name: str
        Name of the Dash component prop
    type_object: dict
        react-docgen-generated prop type dictionary
    required: bool
        Component is required?
    description: str
        Dash component description
    default: dict
        Either None if a default value is not defined, or
        dict containing the key 'value' that defines a
        default value for the prop
    indent_num: int
        Number of indents to use for the context block
        (creates 2 spaces for every indent)
    is_flow_type: bool
        Does the prop use Flow types? Otherwise, uses PropTypes
    Returns
    -------
    str
        Dash component prop docstring
    """
    py_type_name = js_to_py_type(type_object=type_object, is_flow_type=is_flow_type, indent_num=indent_num)
    indent_spacing = '  ' * indent_num
    default = default['value'] if default else ''
    default = fix_keywords(default)
    is_required = 'optional'
    if required:
        is_required = 'required'
    elif default and default not in ['None', '{}', '[]']:
        is_required = 'default ' + default.replace('\n', '')
    period = '.' if description else ''
    description = description.strip().strip('.').replace('"', '\\"') + period
    desc_indent = indent_spacing + '    '
    description = fill(description, initial_indent=desc_indent, subsequent_indent=desc_indent, break_long_words=False, break_on_hyphens=False)
    description = f'\n{description}' if description else ''
    colon = ':' if description else ''
    description = fix_keywords(description)
    if '\n' in py_type_name:
        dict_or_list = 'list of dicts' if py_type_name.startswith('list') else 'dict'
        intro1, intro2, dict_descr = py_type_name.partition('with keys:')
        intro = f'`{prop_name}` is a {intro1}{intro2}'
        intro = fill(intro, initial_indent=desc_indent, subsequent_indent=desc_indent, break_long_words=False, break_on_hyphens=False)
        if '| dict with keys:' in dict_descr:
            dict_part1, dict_part2 = dict_descr.split(' |', 1)
            dict_part2 = ''.join([desc_indent, 'Or', dict_part2])
            dict_descr = f'{dict_part1}\n\n  {dict_part2}'
        current_indent = dict_descr.lstrip('\n').find('-')
        if current_indent == len(indent_spacing):
            dict_descr = ''.join(('\n\n    ' + line for line in dict_descr.splitlines() if line != ''))
        return f'\n{indent_spacing}- {prop_name} ({dict_or_list}; {is_required}){colon}{description}\n\n{intro}{dict_descr}'
    tn = f'{py_type_name}; ' if py_type_name else ''
    return f'\n{indent_spacing}- {prop_name} ({tn}{is_required}){colon}{description}'