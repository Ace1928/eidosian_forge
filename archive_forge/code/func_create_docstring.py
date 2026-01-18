from collections import OrderedDict
import copy
import os
from textwrap import fill, dedent
from dash.development.base_component import _explicitize_args
from dash.exceptions import NonExistentEventException
from ._all_keywords import python_keywords
from ._collect_nodes import collect_nodes, filter_base_nodes
from .base_component import Component
def create_docstring(component_name, props, description, prop_reorder_exceptions=None):
    """Create the Dash component docstring.
    Parameters
    ----------
    component_name: str
        Component name
    props: dict
        Dictionary with {propName: propMetadata} structure
    description: str
        Component description
    Returns
    -------
    str
        Dash component docstring
    """
    props = props if prop_reorder_exceptions is not None and component_name in prop_reorder_exceptions or (prop_reorder_exceptions is not None and 'ALL' in prop_reorder_exceptions) else reorder_props(props)
    n = 'n' if component_name[0].lower() in 'aeiou' else ''
    args = '\n'.join((create_prop_docstring(prop_name=p, type_object=prop['type'] if 'type' in prop else prop['flowType'], required=prop['required'], description=prop['description'], default=prop.get('defaultValue'), indent_num=0, is_flow_type='flowType' in prop and 'type' not in prop) for p, prop in filter_props(props).items()))
    return f'A{n} {component_name} component.\n{description}\n\nKeyword arguments:\n{args}'