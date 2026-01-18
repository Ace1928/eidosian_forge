from __future__ import annotations
import logging # isort:skip
import json
import os
import urllib
from typing import (
from uuid import uuid4
from ..core.types import ID
from ..util.serialization import make_id
from ..util.warnings import warn
from .state import curstate
def load_notebook(resources: Resources | None=None, verbose: bool=False, hide_banner: bool=False, load_timeout: int=5000) -> None:
    """ Prepare the IPython notebook for displaying Bokeh plots.

    Args:
        resources (Resource, optional) :
            how and where to load BokehJS from (default: CDN)

        verbose (bool, optional) :
            whether to report detailed settings (default: False)

        hide_banner (bool, optional):
            whether to hide the Bokeh banner (default: False)

        load_timeout (int, optional) :
            Timeout in milliseconds when plots assume load timed out (default: 5000)

    .. warning::
        Clearing the output cell containing the published BokehJS
        resources HTML code may cause Bokeh CSS styling to be removed.

    Returns:
        None

    """
    global _NOTEBOOK_LOADED
    from .. import __version__
    from ..core.templates import NOTEBOOK_LOAD
    from ..embed.bundle import bundle_for_objs_and_resources
    from ..resources import Resources
    from ..settings import settings
    from ..util.serialization import make_globally_unique_css_safe_id
    if resources is None:
        resources = Resources(mode=settings.resources())
    element_id: ID | None
    html: str | None
    if not hide_banner:
        if resources.mode == 'inline':
            js_info: str | list[str] = 'inline'
            css_info: str | list[str] = 'inline'
        else:
            js_info = resources.js_files[0] if len(resources.js_files) == 1 else resources.js_files
            css_info = resources.css_files[0] if len(resources.css_files) == 1 else resources.css_files
        warnings = ['Warning: ' + msg.text for msg in resources.messages if msg.type == 'warn']
        if _NOTEBOOK_LOADED and verbose:
            warnings.append('Warning: BokehJS previously loaded')
        element_id = make_globally_unique_css_safe_id()
        html = NOTEBOOK_LOAD.render(element_id=element_id, verbose=verbose, js_info=js_info, css_info=css_info, bokeh_version=__version__, warnings=warnings)
    else:
        element_id = None
        html = None
    _NOTEBOOK_LOADED = resources
    bundle = bundle_for_objs_and_resources(None, resources)
    nb_js = _loading_js(bundle, element_id, load_timeout, register_mime=True)
    jl_js = _loading_js(bundle, element_id, load_timeout, register_mime=False)
    if html is not None:
        publish_display_data({'text/html': html})
    publish_display_data({JS_MIME_TYPE: nb_js, LOAD_MIME_TYPE: jl_js})