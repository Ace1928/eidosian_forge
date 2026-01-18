import io
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from cProfile import Profile
from functools import wraps
from ..config import config
from ..util import escape
from .state import state
def profiling_tabs(state, allow=None, deny=[]):
    from ..layout import Accordion, Column, Row, Tabs
    from ..widgets import Checkbox, Select
    tabs = Tabs(*get_profiles(get_sessions(allow, deny)), margin=(0, 5), sizing_mode='stretch_width')

    def update_profiles(*args, **kwargs):
        tabs[:] = get_profiles(get_sessions(allow, deny), **kwargs)
    state.param.watch(update_profiles, '_profiles')
    if config.profiler == 'pyinstrument':

        def update_pyinstrument(*args):
            update_profiles(timeline=timeline.value, show_all=show_all.value)
        timeline = Checkbox(name='Enable timeline', margin=(5, 0))
        timeline.param.watch(update_pyinstrument, 'value')
        show_all = Checkbox(name='Show All', margin=(5, 0))
        show_all.param.watch(update_pyinstrument, 'value')
        config_panel = Row(timeline, show_all, sizing_mode='stretch_width')
    elif config.profiler == 'memray':

        def update_memray(*args):
            update_profiles(reporter=reporter.value)
        reporter = Select(name='Reporter', options=['flamegraph', 'table', 'tree'], value='tree')
        reporter.param.watch(update_memray, 'value')
        config_panel = reporter
    else:
        config_panel = Row(sizing_mode='stretch_width')
    return Column(Accordion(('Config', config_panel), active=[], active_header_background='#444444', header_background='#333333', sizing_mode='stretch_width', margin=0), tabs, sizing_mode='stretch_width')