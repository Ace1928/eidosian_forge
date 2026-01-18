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
def get_session_data():
    durations, renders, sessions = ([], [], [])
    session_info = state.session_info['sessions']
    for i, session in enumerate(session_info.values()):
        is_live = session['ended'] is None
        live = sum([1 for s in session_info.values() if s['launched'] < session['launched'] and (not s['ended'] or s['ended'] > session['launched'])]) + 1
        if session['rendered'] is not None:
            renders.append(session['rendered'] - session['started'])
        if not is_live:
            durations.append(session['ended'] - session['launched'])
        duration = np.mean(durations) if durations else 0
        render = np.mean(renders) if renders else 0
        sessions.append((session['launched'], live, i + 1, render, duration))
    if not sessions:
        i = -1
        duration = 0
        render = 0
    now = dt.datetime.now().timestamp()
    live = sum([1 for s in session_info.values() if s['launched'] < now and (not s['ended'] or s['ended'] > now)])
    sessions.append((now, live, i + 1, render, duration))
    return pd.DataFrame(sessions, columns=['time', 'live', 'total', 'render', 'duration'])