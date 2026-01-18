import os
from typing import Dict
from ...utils.mimebundle import spec_to_mimebundle
from ..display import (
from .schema import SCHEMA_VERSION
from typing import Final
def svg_renderer(spec: dict, **metadata) -> Dict[str, str]:
    return spec_to_mimebundle(spec, format='svg', mode='vega-lite', vega_version=VEGA_VERSION, vegaembed_version=VEGAEMBED_VERSION, vegalite_version=VEGALITE_VERSION, **metadata)