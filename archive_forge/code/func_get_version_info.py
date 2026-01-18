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
def get_version_info():
    from panel import __version__
    return HTML(f'\n    <h4>\n    Panel Server running on following versions:\n    </h4>\n    <code>\n    Python {sys.version.split('|')[0]}</br>\n    Panel: {__version__}</br>\n    Bokeh: {bokeh.__version__}</br>\n    Param: {param.__version__}</br>\n    </code>', width=300, height=300, margin=(0, 5))