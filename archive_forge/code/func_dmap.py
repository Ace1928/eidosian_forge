from a list that contains the shared attribute. This is required for the
import abc
import operator
import sys
from functools import partial
from packaging.version import Version
from types import FunctionType, MethodType
import holoviews as hv
import pandas as pd
import panel as pn
import param
from panel.layout import Column, Row, VSpacer, HSpacer
from panel.util import get_method_owner, full_groupby
from panel.widgets.base import Widget
from .converter import HoloViewsConverter
from .util import (
def dmap(self):
    """
        Wraps the output in a DynamicMap. Only valid if the output
        is a HoloViews object.
        """
    return hv.DynamicMap(self._callback)