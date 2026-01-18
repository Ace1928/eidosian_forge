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
def log_component():
    log_terminal.param.trigger('value')
    return Column(Accordion(('Filters & Download', Row(level_filter, app_filter, session_filter, message_filter, Column(download_filename, download_button, reset_filter), sizing_mode='stretch_width')), active=[], active_header_background='#444444', header_background='#333333', sizing_mode='stretch_width'), log_terminal, sizing_mode='stretch_both')