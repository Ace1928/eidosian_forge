import os
from typing import Dict
from ...utils.mimebundle import spec_to_mimebundle
from ..display import (
from .schema import SCHEMA_VERSION
from typing import Final
def browser_renderer(spec: dict, offline=False, using=None, port=0, **metadata) -> Dict[str, str]:
    from altair.utils._show import open_html_in_browser
    if offline:
        metadata['template'] = 'inline'
    mimebundle = spec_to_mimebundle(spec, format='html', mode='vega-lite', vega_version=VEGA_VERSION, vegaembed_version=VEGAEMBED_VERSION, vegalite_version=VEGALITE_VERSION, **metadata)
    if isinstance(mimebundle, tuple):
        mimebundle = mimebundle[0]
    html = mimebundle['text/html']
    open_html_in_browser(html, using=using, port=port)
    return {}