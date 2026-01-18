import functools
import inspect
import json
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, cast
from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel
def get_fn_name(fn: Callable):
    """Returns the name of a callable."""
    if not callable(fn):
        raise TypeError('The `name` filter only applies to callables.')
    if not hasattr(fn, '__name__'):
        name = type(fn).__name__
    else:
        name = fn.__name__
    return name