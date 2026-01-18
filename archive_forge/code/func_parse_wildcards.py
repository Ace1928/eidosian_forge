from collections import OrderedDict
import copy
import os
from textwrap import fill, dedent
from dash.development.base_component import _explicitize_args
from dash.exceptions import NonExistentEventException
from ._all_keywords import python_keywords
from ._collect_nodes import collect_nodes, filter_base_nodes
from .base_component import Component
def parse_wildcards(props):
    """Pull out the wildcard attributes from the Component props.
    Parameters
    ----------
    props: dict
        Dictionary with {propName: propMetadata} structure
    Returns
    -------
    list
        List of Dash valid wildcard prefixes
    """
    list_of_valid_wildcard_attr_prefixes = []
    for wildcard_attr in ['data-*', 'aria-*']:
        if wildcard_attr in props:
            list_of_valid_wildcard_attr_prefixes.append(wildcard_attr[:-1])
    return list_of_valid_wildcard_attr_prefixes