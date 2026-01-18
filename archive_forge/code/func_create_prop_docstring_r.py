import os
import sys
import shutil
import importlib
import textwrap
import re
import warnings
from ._all_keywords import r_keywords
from ._py_components_generation import reorder_props
def create_prop_docstring_r(prop_name, type_object, required, description, indent_num, is_flow_type=False):
    """
    Create the Dash component prop docstring
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
    r_type_name = get_r_type(type_object=type_object, is_flow_type=is_flow_type, indent_num=indent_num + 1)
    indent_spacing = '  ' * indent_num
    if '\n' in r_type_name:
        return '{indent_spacing}- {name} ({is_required}): {description}. {name} has the following type: {type}'.format(indent_spacing=indent_spacing, name=prop_name, type=r_type_name, description=description, is_required='required' if required else 'optional')
    return '{indent_spacing}- {name} ({type}{is_required}){description}'.format(indent_spacing=indent_spacing, name=prop_name, type='{}; '.format(r_type_name) if r_type_name else '', description=': {}'.format(description) if description != '' else '', is_required='required' if required else 'optional')