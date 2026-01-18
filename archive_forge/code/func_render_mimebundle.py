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
def render_mimebundle(model: 'Model', doc: 'Document', comm: 'Comm', manager: Optional['CommManager']=None, location: Optional['Location']=None, resources: str='cdn') -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """
    Displays bokeh output inside a notebook using the PyViz display
    and comms machinery.
    """
    if manager and manager.client_comm_id in _JupyterCommManager._comms:
        client_comm = _JupyterCommManager._comms[manager.client_comm_id]
        if not client_comm._on_open:
            client_comm._on_open = lambda _: comm.init()
    if not isinstance(model, Model):
        raise ValueError('Can only render bokeh LayoutDOM models')
    add_to_doc(model, doc, True)
    if manager is not None:
        doc.add_root(manager)
    if location is not None:
        loc = location._get_model(doc, model, model, comm)
        doc.add_root(loc)
    return render_model(model, comm, resources)