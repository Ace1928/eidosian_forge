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
def show_server(panel: Any, notebook_url: str, port: int=0) -> 'Server':
    """
    Displays a bokeh server inline in the notebook.

    Arguments
    ---------
    panel: Viewable
      Panel Viewable object to launch a server for
    notebook_url: str
      The URL of the running Jupyter notebook server
    port: int (optional, default=0)
      Allows specifying a specific port
    server_id: str
      Unique ID to identify the server with

    Returns
    -------
    server: bokeh.server.Server
    """
    from IPython.display import publish_display_data
    from .server import _origin_url, _server_url, get_server
    if callable(notebook_url):
        origin = notebook_url(None)
    else:
        origin = _origin_url(notebook_url)
    server_id = uuid.uuid4().hex
    server = get_server(panel, port=port, websocket_origin=origin, start=True, show=False, server_id=server_id)
    if callable(notebook_url):
        url = notebook_url(server.port)
    else:
        url = _server_url(notebook_url, server.port)
    script = server_document(url, resources=None)
    publish_display_data({HTML_MIME: script, EXEC_MIME: ''}, metadata={EXEC_MIME: {'server_id': server_id}})
    return server