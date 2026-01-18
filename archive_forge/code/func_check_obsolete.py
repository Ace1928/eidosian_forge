import sys
from collections.abc import MutableSequence
import re
from textwrap import dedent
from keyword import iskeyword
import flask
from ._grouping import grouping_len, map_grouping
from .development.base_component import Component
from . import exceptions
from ._utils import (
def check_obsolete(kwargs):
    for key in kwargs:
        if key in ['components_cache_max_age', 'static_folder']:
            raise exceptions.ObsoleteKwargException(f'\n                {key} is no longer a valid keyword argument in Dash since v1.0.\n                See https://dash.plotly.com for details.\n                ')
        if key in ['dynamic_loading', 'preloaded_libraries']:
            print(f'{key} has been removed and no longer a valid keyword argument in Dash.', file=sys.stderr)
            continue
        raise TypeError(f"Dash() got an unexpected keyword argument '{key}'")