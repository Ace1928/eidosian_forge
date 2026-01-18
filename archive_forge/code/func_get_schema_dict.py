import functools
import inspect
import json
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, cast
from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel
@get_schema.register(dict)
def get_schema_dict(model: Dict):
    """Return a pretty-printed dictionary"""
    return json.dumps(model, indent=2)