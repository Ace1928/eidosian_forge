import ast
import copy
import importlib
import inspect
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from weakref import WeakKeyDictionary
import param
from bokeh.core.has_props import _default_resolver
from bokeh.document import Document
from bokeh.model import Model
from bokeh.settings import settings as bk_settings
from pyviz_comms import (
from .io.logging import panel_log_handler
from .io.state import state
from .util import param_watchers
@property
def oauth_extra_params(self):
    if 'PANEL_OAUTH_EXTRA_PARAMS' in os.environ:
        return ast.literal_eval(os.environ['PANEL_OAUTH_EXTRA_PARAMS'])
    else:
        return self._oauth_extra_params