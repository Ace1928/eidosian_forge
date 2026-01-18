import json
import pkgutil
import textwrap
from typing import Callable, Dict, Optional, Tuple, Any, Union
import uuid
from ._vegafusion_data import compile_with_vegafusion, using_vegafusion
from .plugin_registry import PluginRegistry, PluginEnabler
from .mimebundle import spec_to_mimebundle
from .schemapi import validate_jsonschema
def json_renderer_base(spec: dict, str_repr: str, **options) -> DefaultRendererReturnType:
    """A renderer that returns a MIME type of application/json.

    In JupyterLab/nteract this is rendered as a nice JSON tree.
    """
    return default_renderer_base(spec, mime_type='application/json', str_repr=str_repr, **options)