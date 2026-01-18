import datetime as dt
import logging
import os
import sys
import time
from functools import partial
import bokeh
import numpy as np
import pandas as pd
import param
from bokeh.models import HoverTool
from bokeh.plotting import ColumnDataSource, figure
from ..config import config, panel_extension as extension
from ..depends import bind
from ..layout import (
from ..pane import HTML, Bokeh
from ..template import FastListTemplate
from ..widgets import (
from ..widgets.indicators import Trend
from .logging import (
from .notebook import push_notebook
from .profile import profiling_tabs
from .server import set_curdoc
from .state import state
def get_overview(doc=None):
    layout = FlexBox(*get_session_info(doc), margin=0, sizing_mode='stretch_width')
    info = get_version_info()
    try:
        import psutil
    except Exception:
        layout.append(info)
        return layout
    else:
        layout.extend([*get_process_info(), info])
        return layout