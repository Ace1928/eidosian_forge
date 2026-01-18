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
def render_snakeviz(name, sessions):
    from pstats import Stats
    import snakeviz
    from snakeviz.stats import json_stats, table_rows
    from tornado.template import Template
    SNAKEVIZ_PATH = os.path.join(os.path.dirname(snakeviz.__file__), 'templates', 'viz.html')
    with open(SNAKEVIZ_PATH) as f:
        SNAKEVIZ_TEMPLATE = Template(f.read())
    pstats = Stats(sessions[0])
    for session in sessions[1:]:
        pstats.add(session)
    rendered = SNAKEVIZ_TEMPLATE.generate(profile_name=name, table_rows=table_rows(pstats), callees=json_stats(pstats)).decode('utf-8').replace('/static/', '/snakeviz/static/')
    return (escape(rendered), 'background-color: white;')