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
def record_events(doc):
    msg = diff(doc, binary=False)
    if msg is None:
        return {'header': '{}', 'metadata': '{}', 'content': '{}'}
    return {'header': msg.header_json, 'metadata': msg.metadata_json, 'content': msg.content_json}