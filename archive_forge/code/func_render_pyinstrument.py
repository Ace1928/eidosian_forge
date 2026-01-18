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
def render_pyinstrument(sessions, timeline=False, show_all=False):
    from pyinstrument.renderers import HTMLRenderer
    from pyinstrument.session import Session
    r = HTMLRenderer(timeline=timeline, show_all=show_all)
    if timeline:
        session = sessions[-1]
    else:
        session = sessions[0]
        for s in sessions[1:]:
            session = Session.combine(session, s)
    try:
        rendered = r.render(session)
    except Exception:
        rendered = '<h2><b>Rendering pyinstrument session failed</b></h2>'
    return (escape(rendered), '')