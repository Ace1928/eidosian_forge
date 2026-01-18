from __future__ import annotations
import json
import os
import sys
import uuid
import warnings
from contextlib import contextmanager
from functools import partial
from typing import (
import bokeh
import bokeh.embed.notebook
import param
from bokeh.core.json_encoder import serialize_json
from bokeh.core.templates import MACROS
from bokeh.document import Document
from bokeh.embed import server_document
from bokeh.embed.elements import div_for_render_item, script_for_render_items
from bokeh.embed.util import standalone_docs_json_and_render_items
from bokeh.embed.wrappers import wrap_in_script_tag
from bokeh.models import Model
from bokeh.resources import CDN, INLINE
from bokeh.settings import _Unset, settings
from bokeh.util.serialization import make_id
from param.display import (
from pyviz_comms import (
from ..util import escape
from .embed import embed_state
from .model import add_to_doc, diff
from .resources import (
from .state import state
def require_components():
    """
    Returns JS snippet to load the required dependencies in the classic
    notebook using REQUIRE JS.
    """
    from ..config import config
    configs, requirements, exports = ([], [], {})
    js_requires = []
    for qual_name, model in Model.model_class_reverse_map.items():
        if '.' in qual_name:
            js_requires.append(model)
    from ..reactive import ReactiveHTML
    js_requires += list(param.concrete_descendents(ReactiveHTML).values())
    for export, js in config.js_files.items():
        name = js.split('/')[-1].replace('.min', '').split('.')[-2]
        conf = {'paths': {name: js[:-3]}, 'exports': {name: export}}
        js_requires.append(conf)
    skip_import = {}
    for model in js_requires:
        if not isinstance(model, dict) and issubclass(model, ReactiveHTML) and (not model._loaded()):
            continue
        if hasattr(model, '__js_skip__'):
            skip_import.update(model.__js_skip__)
        if not (hasattr(model, '__js_require__') or isinstance(model, dict)):
            continue
        if isinstance(model, dict):
            model_require = model
        else:
            model_require = dict(model.__js_require__)
        model_exports = model_require.pop('exports', {})
        if not any((model_require == config for config in configs)):
            configs.append(model_require)
        for req in list(model_require.get('paths', [])):
            if isinstance(req, tuple):
                model_require['paths'] = dict(model_require['paths'])
                model_require['paths'][req[0]] = model_require['paths'].pop(req)
            reqs = req[1] if isinstance(req, tuple) else (req,)
            for r in reqs:
                if r not in requirements:
                    requirements.append(r)
                    if r in model_exports:
                        exports[r] = model_exports[r]
    return (configs, requirements, exports, skip_import)