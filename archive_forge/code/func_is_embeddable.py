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
def is_embeddable(object):
    if not isinstance(object, Widget) or object.disabled:
        return False
    if isinstance(object, DiscreteSlider):
        return ref in object._composite[1]._models
    return ref in object._models