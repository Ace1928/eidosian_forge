import json
import os
import sys
import uuid
from collections import defaultdict
from contextlib import contextmanager
from itertools import product
import param
from bokeh.core.property.bases import Property
from bokeh.models import CustomJS
from param.parameterized import Watcher
from ..util import param_watchers
from .model import add_to_doc, diff
from .state import state
def param_to_jslink(model, widget):
    """
    Converts Param pane widget links into JS links if possible.
    """
    from ..reactive import Reactive
    from ..widgets import LiteralInput, Widget
    param_pane = widget._param_pane
    pobj = param_pane.object
    pname = [k for k, v in param_pane._widgets.items() if v is widget]
    watchers = [w for w in get_watchers(widget) if w not in widget._internal_callbacks and w not in param_pane._internal_callbacks]
    if isinstance(pobj, Reactive):
        tgt_links = [Watcher(*l[:-4]) for l in pobj._links]
        tgt_watchers = [w for w in get_watchers(pobj) if w not in pobj._internal_callbacks and w not in tgt_links and (w not in param_pane._internal_callbacks)]
    else:
        tgt_watchers = []
    for widget in param_pane._widgets.values():
        if isinstance(widget, LiteralInput):
            widget.serializer = 'json'
    if not pname or not isinstance(pobj, Reactive) or watchers or (pname[0] not in pobj._linkable_params) or (not isinstance(pobj, Widget) and tgt_watchers):
        return
    return link_to_jslink(model, widget, 'value', pobj, pname[0])