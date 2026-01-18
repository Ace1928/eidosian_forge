import os
from typing import Dict
from ...utils.mimebundle import spec_to_mimebundle
from ..display import (
from .schema import SCHEMA_VERSION
from typing import Final
def mimetype_renderer(spec: dict, **metadata) -> DefaultRendererReturnType:
    return default_renderer_base(spec, VEGALITE_MIME_TYPE, DEFAULT_DISPLAY, **metadata)