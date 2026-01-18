import base64
import os
from contextlib import contextmanager
from functools import partial
from io import BytesIO, StringIO
import panel as pn
import param
from bokeh.document import Document
from bokeh.embed import file_html
from bokeh.io import curdoc
from bokeh.resources import CDN, INLINE
from packaging.version import Version
from panel import config
from panel.io.notebook import ipywidget, load_notebook, render_mimebundle, render_model
from panel.io.state import state
from panel.models.comm_manager import CommManager as PnCommManager
from panel.pane import HoloViews as HoloViewsPane
from panel.viewable import Viewable
from panel.widgets.player import PlayerBase
from pyviz_comms import CommManager
from param.parameterized import bothmethod
from ..core import AdjointLayout, DynamicMap, HoloMap, Layout
from ..core.data import disable_pipeline
from ..core.io import Exporter
from ..core.options import Compositor, SkipRendering, Store, StoreOptions
from ..core.util import unbound_dimensions
from ..streams import Stream
from . import Plot
from .util import collate, displayable, initialize_dynamic
@bothmethod
def export_widgets(self_or_cls, obj, filename, fmt=None, template=None, json=False, json_path='', **kwargs):
    """
        Render and export object as a widget to a static HTML
        file. Allows supplying a custom template formatting string
        with fields to interpolate 'js', 'css' and the main 'html'
        containing the widget. Also provides options to export widget
        data to a json file in the supplied json_path (defaults to
        current path).
        """
    if fmt not in self_or_cls.widgets + ['auto', None]:
        raise ValueError('Renderer.export_widget may only export registered widget types.')
    self_or_cls.get_widget(obj, fmt).save(filename)