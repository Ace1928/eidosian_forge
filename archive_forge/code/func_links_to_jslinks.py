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
def links_to_jslinks(model, widget):
    from ..widgets import Widget
    src_links = [Watcher(*l[:-4]) for l in widget._links]
    if any((w not in widget._internal_callbacks and w not in src_links for w in get_watchers(widget))):
        return
    links = []
    for link in widget._links:
        target = link.target
        tgt_watchers = [w for w in get_watchers(target) if w not in target._internal_callbacks]
        if link.transformed or (tgt_watchers and (not isinstance(target, Widget))):
            return
        mappings = []
        for pname, tgt_spec in link.links.items():
            if Watcher(*link[:-4]) in param_watchers(widget)[pname]['value']:
                mappings.append((pname, tgt_spec))
        if mappings:
            links.append((link, mappings))
    jslinks = []
    for link, mapping in links:
        for src_spec, tgt_spec in mapping:
            jslink = link_to_jslink(model, widget, src_spec, link.target, tgt_spec)
            if jslink is None:
                return
            widget.param.trigger(src_spec)
            jslinks.append(jslink)
    return jslinks