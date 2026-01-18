from collections import OrderedDict
import copy
import os
from textwrap import fill, dedent
from dash.development.base_component import _explicitize_args
from dash.exceptions import NonExistentEventException
from ._all_keywords import python_keywords
from ._collect_nodes import collect_nodes, filter_base_nodes
from .base_component import Component
def required_props(props):
    """Pull names of required props from the props object.
    Parameters
    ----------
    props: dict
    Returns
    -------
    list
        List of prop names (str) that are required for the Component
    """
    return [prop_name for prop_name, prop in list(props.items()) if prop['required']]