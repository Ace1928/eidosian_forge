from contextlib import contextmanager
import errno
from io import BytesIO
import json
import mimetypes
from pathlib import Path
import random
import sys
import signal
import threading
import tornado.web
import tornado.ioloop
import tornado.websocket
import matplotlib as mpl
from matplotlib.backend_bases import _Backend
from matplotlib._pylab_helpers import Gcf
from . import backend_webagg_core as core
from .backend_webagg_core import (  # noqa: F401 # pylint: disable=W0611
def ipython_inline_display(figure):
    import tornado.template
    WebAggApplication.initialize()
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        if not webagg_server_thread.is_alive():
            webagg_server_thread.start()
    fignum = figure.number
    tpl = Path(core.FigureManagerWebAgg.get_static_file_path(), 'ipython_inline_figure.html').read_text()
    t = tornado.template.Template(tpl)
    return t.generate(prefix=WebAggApplication.url_prefix, fig_id=fignum, toolitems=core.NavigationToolbar2WebAgg.toolitems, canvas=figure.canvas, port=WebAggApplication.port).decode('utf-8')