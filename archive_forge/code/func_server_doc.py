from __future__ import annotations
import os
import sys
import uuid
from functools import partial
from pathlib import Path, PurePath
from typing import (
import jinja2
import param
from bokeh.document.document import Document
from bokeh.models import LayoutDOM
from bokeh.settings import settings as _settings
from pyviz_comms import JupyterCommManager as _JupyterCommManager
from ..config import _base_config, config, panel_extension
from ..io.document import init_doc
from ..io.model import add_to_doc
from ..io.notebook import render_template
from ..io.notifications import NotificationArea
from ..io.resources import (
from ..io.save import save
from ..io.state import curdoc_locked, state
from ..layout import Column, GridSpec, ListLike
from ..models.comm_manager import CommManager
from ..pane import (
from ..pane.image import ImageBase
from ..reactive import ReactiveHTML
from ..theme.base import (
from ..util import isurl
from ..viewable import (
from ..widgets import Button
from ..widgets.indicators import BooleanIndicator, LoadingSpinner
def server_doc(self, doc: Optional[Document]=None, title: str=None, location: bool | Location=True) -> Document:
    """
        Returns a servable bokeh Document with the panel attached

        Arguments
        ---------
        doc : bokeh.Document (optional)
          The Bokeh Document to attach the panel to as a root,
          defaults to bokeh.io.curdoc()
        title : str
          A string title to give the Document
        location : boolean or panel.io.location.Location
          Whether to create a Location component to observe and
          set the URL location.

        Returns
        -------
        doc : bokeh.Document
          The Bokeh document the panel was attached to
        """
    return self._init_doc(doc, title=title, location=location)