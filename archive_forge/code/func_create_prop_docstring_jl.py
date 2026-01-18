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
def create_prop_docstring_jl(prop_name, type_object, required, description, indent_num):
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
    jl_type_name = get_jl_type(type_object=type_object)
    indent_spacing = '  ' * indent_num
    if '\n' in jl_type_name:
        return '{indent_spacing}- `{name}` ({is_required}): {description}. {name} has the following type: {type}'.format(indent_spacing=indent_spacing, name=prop_name, type=jl_type_name, description=description, is_required='required' if required else 'optional')
    return '{indent_spacing}- `{name}` ({type}{is_required}){description}'.format(indent_spacing=indent_spacing, name=prop_name, type='{}; '.format(jl_type_name) if jl_type_name else '', description=': {}'.format(description) if description != '' else '', is_required='required' if required else 'optional')