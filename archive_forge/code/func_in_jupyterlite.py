import asyncio
import sys
from bokeh.document import Document
from bokeh.embed.elements import script_for_render_items
from bokeh.embed.util import standalone_docs_json_and_render_items
from bokeh.embed.wrappers import wrap_in_script_tag
from panel.io.pyodide import _link_docs
from panel.pane import panel as as_panel
from .core.dimension import LabelledData
from .core.options import Store
from .util import extension as _extension
def in_jupyterlite():
    import js
    return hasattr(js, '_JUPYTERLAB') or hasattr(js, 'webpackChunk_jupyterlite_pyodide_kernel_extension') or (not hasattr(js, 'document'))