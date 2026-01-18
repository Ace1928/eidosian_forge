from collections import OrderedDict
import copy
import os
from textwrap import fill, dedent
from dash.development.base_component import _explicitize_args
from dash.exceptions import NonExistentEventException
from ._all_keywords import python_keywords
from ._collect_nodes import collect_nodes, filter_base_nodes
from .base_component import Component
def prohibit_events(props):
    """Events have been removed. Raise an error if we see dashEvents or
    fireEvents.
    Parameters
    ----------
    props: dict
        Dictionary with {propName: propMetadata} structure
    Raises
    -------
    ?
    """
    if 'dashEvents' in props or 'fireEvents' in props:
        raise NonExistentEventException('Events are no longer supported by dash. Use properties instead, eg `n_clicks` instead of a `click` event.')