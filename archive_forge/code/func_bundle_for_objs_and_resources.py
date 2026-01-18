from __future__ import annotations
import logging # isort:skip
import hashlib
import json
import os
from dataclasses import dataclass
from os.path import normpath
from pathlib import Path
from typing import (
from urllib.parse import urljoin
from ..core.has_props import HasProps
from ..core.templates import CSS_RESOURCES, JS_RESOURCES
from ..document.document import Document
from ..resources import Resources
from ..settings import settings
from ..util.compiler import bundle_models
from .util import contains_tex_string
def bundle_for_objs_and_resources(objs: Sequence[HasProps | Document] | None, resources: Resources | None) -> Bundle:
    """ Generate rendered CSS and JS resources suitable for the given
    collection of Bokeh objects

    Args:
        objs (seq[HasProps or Document]) :

        resources (Resources)

    Returns:
        Bundle

    """
    if objs is not None:
        all_objs = _all_objs(objs)
        use_widgets = _use_widgets(all_objs)
        use_tables = _use_tables(all_objs)
        use_gl = _use_gl(all_objs)
        use_mathjax = _use_mathjax(all_objs)
    else:
        all_objs = None
        use_widgets = True
        use_tables = True
        use_gl = True
        use_mathjax = True
    js_files: list[URL] = []
    js_raw: list[str] = []
    css_files: list[URL] = []
    css_raw: list[str] = []
    if resources is not None:
        components = list(resources.components)
        if not use_widgets:
            components.remove('bokeh-widgets')
        if not use_tables:
            components.remove('bokeh-tables')
        if not use_gl:
            components.remove('bokeh-gl')
        if not use_mathjax:
            components.remove('bokeh-mathjax')
        resources = resources.clone(components=components)
        js_files.extend(map(URL, resources.js_files))
        js_raw.extend(resources.js_raw)
        css_files.extend(map(URL, resources.css_files))
        css_raw.extend(resources.css_raw)
        extensions = _bundle_extensions(all_objs if objs else None, resources)
        mode = resources.mode
        if mode == 'inline':
            js_raw.extend([Resources._inline(bundle.artifact_path) for bundle in extensions])
        elif mode == 'server':
            js_files.extend([bundle.server_url for bundle in extensions])
        elif mode == 'cdn':
            for bundle in extensions:
                if bundle.cdn_url is not None:
                    js_files.append(bundle.cdn_url)
                else:
                    js_raw.append(Resources._inline(bundle.artifact_path))
        else:
            js_files.extend([URL(str(bundle.artifact_path)) for bundle in extensions])
    models = [obj.__class__ for obj in all_objs] if all_objs else None
    ext = bundle_models(models)
    if ext is not None:
        js_raw.append(ext)
    return Bundle(js_files, js_raw, css_files, css_raw, resources.hashes if resources else {})