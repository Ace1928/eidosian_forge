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
def servable(self, title: Optional[str]=None, location: bool | 'Location'=True, area: str='main', target: Optional[str]=None) -> 'BaseTemplate':
    """
        Serves the template and returns self to allow it to display
        itself in a notebook context.

        Arguments
        ---------
        title : str
          A string title to give the Document (if served as an app)
        location : boolean or panel.io.location.Location
          Whether to create a Location component to observe and
          set the URL location.
        area: str (deprecated)
          The area of a template to add the component too. Only has an
          effect if pn.config.template has been set.
        target: str
          Target area to write to. If a template has been configured
          on pn.config.template this refers to the target area in the
          template while in pyodide this refers to the ID of the DOM
          node to write to.

        Returns
        -------
        The template
        """
    if curdoc_locked().session_context and config.template:
        raise RuntimeError('Cannot mark template as servable if a global template is defined. Either explicitly construct a template and serve it OR set `pn.config.template`, whether directly or via `pn.extension(template=...)`, not both.')
    return super().servable(title, location, area, target)