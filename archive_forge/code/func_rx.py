from __future__ import annotations
import math
from typing import (
import param  # type: ignore
from bokeh.models import ImportedStyleSheet, Tooltip
from bokeh.models.dom import HTML
from param.parameterized import register_reference_transform
from .._param import Margin
from ..layout.base import Row
from ..reactive import Reactive
from ..viewable import Layoutable, Viewable
@property
def rx(self):
    return self.param.value.rx