import calendar
import datetime as dt
import re
import time
from collections import defaultdict
from contextlib import contextmanager, suppress
from itertools import permutations
import bokeh
import numpy as np
import pandas as pd
from bokeh.core.json_encoder import serialize_json  # noqa (API import)
from bokeh.core.property.datetime import Datetime
from bokeh.core.validation import silence
from bokeh.layouts import Column, Row, group_tools
from bokeh.models import (
from bokeh.models.formatters import PrintfTickFormatter, TickFormatter
from bokeh.models.scales import CategoricalScale, LinearScale, LogScale
from bokeh.models.widgets import DataTable, Div
from bokeh.plotting import figure
from bokeh.themes import built_in_themes
from bokeh.themes.theme import Theme
from packaging.version import Version
from ...core.layout import Layout
from ...core.ndmapping import NdMapping
from ...core.overlay import NdOverlay, Overlay
from ...core.spaces import DynamicMap, get_nested_dmaps
from ...core.util import (
from ...util.warnings import warn
from ..util import dim_axis_label
def recursive_model_update(model, props):
    """
    Recursively updates attributes on a model including other
    models. If the type of the new model matches the old model
    properties are simply updated, otherwise the model is replaced.
    """
    updates = {}
    valid_properties = model.properties_with_values()
    for k, v in props.items():
        if isinstance(v, Model):
            nested_model = getattr(model, k)
            if type(v) is type(nested_model):
                nested_props = v.properties_with_values(include_defaults=False)
                recursive_model_update(nested_model, nested_props)
            else:
                try:
                    setattr(model, k, v)
                except Exception as e:
                    if isinstance(v, dict) and 'value' in v:
                        setattr(model, k, v['value'])
                    else:
                        raise e
        elif k in valid_properties and v != valid_properties[k]:
            if isinstance(v, dict) and 'value' in v:
                v = v['value']
            updates[k] = v
    model.update(**updates)