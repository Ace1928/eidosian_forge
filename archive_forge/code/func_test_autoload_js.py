import asyncio
import datetime as dt
import logging
import os
import pathlib
import time
import weakref
import param
import pytest
import requests
from bokeh.events import ButtonClick
from panel.config import config
from panel.io import state
from panel.io.resources import DIST_DIR, JS_VERSION
from panel.io.server import INDEX_HTML, get_server, set_curdoc
from panel.layout import Row
from panel.models import HTML as BkHTML
from panel.models.tabulator import TableEditEvent
from panel.pane import Markdown
from panel.param import ParamFunction
from panel.reactive import ReactiveHTML
from panel.template import BootstrapTemplate
from panel.tests.util import serve_and_request, serve_and_wait, wait_until
from panel.widgets import (
def test_autoload_js(port):
    html = Markdown('# Title')
    app_name = 'test'
    args = f'bokeh-autoload-element=1002&bokeh-app-path=/{app_name}&bokeh-absolute-url=http://localhost:{port}/{app_name}'
    r = serve_and_request({app_name: html}, port=port, suffix=f'/{app_name}/autoload.js?{args}')
    assert r.status_code == 200
    assert f'http://localhost:{port}/static/extensions/panel/panel.min.js' in r.content.decode('utf-8')