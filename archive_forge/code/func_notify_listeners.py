import ipywidgets as widgets
import pandas as pd
import numpy as np
import json
from types import FunctionType
from IPython.display import display
from numbers import Integral
from traitlets import (
from itertools import chain
from uuid import uuid4
from six import string_types
from distutils.version import LooseVersion
def notify_listeners(self, event, qgrid_widget):
    event_listeners = self._listeners.get(event['name'], [])
    all_listeners = self._listeners.get(All, [])
    for c in chain(event_listeners, all_listeners):
        c(event, qgrid_widget)