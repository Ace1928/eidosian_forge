import json
import pkgutil
import textwrap
from typing import Callable, Dict, Optional, Tuple, Any, Union
import uuid
from ._vegafusion_data import compile_with_vegafusion, using_vegafusion
from .plugin_registry import PluginRegistry, PluginEnabler
from .mimebundle import spec_to_mimebundle
from .schemapi import validate_jsonschema
def default_renderer_base(spec: dict, mime_type: str, str_repr: str, **options) -> DefaultRendererReturnType:
    """A default renderer for Vega or VegaLite that works for modern frontends.

    This renderer works with modern frontends (JupyterLab, nteract) that know
    how to render the custom VegaLite MIME type listed above.
    """
    from altair.vegalite.v5.display import VEGA_MIME_TYPE, VEGALITE_MIME_TYPE
    assert isinstance(spec, dict)
    bundle: Dict[str, Union[str, dict]] = {}
    metadata: Dict[str, Dict[str, Any]] = {}
    if using_vegafusion():
        spec = compile_with_vegafusion(spec)
        if mime_type == VEGALITE_MIME_TYPE:
            mime_type = VEGA_MIME_TYPE
    bundle[mime_type] = spec
    bundle['text/plain'] = str_repr
    if options:
        metadata[mime_type] = options
    return (bundle, metadata)