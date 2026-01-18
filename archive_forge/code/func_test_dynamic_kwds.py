import numpy as np
import pandas as pd
import pytest
from unittest import TestCase, SkipTest
from hvplot.util import (
def test_dynamic_kwds(self):
    import panel as pn
    from ..util import process_dynamic_args
    x = 'sepal_length'
    y = 'sepal_width'
    kind = 'scatter'
    color = pn.widgets.ColorPicker(value='#ff0000')
    dynamic, arg_deps, arg_names = process_dynamic_args(x, y, kind, c=color)
    assert 'x' not in dynamic
    assert 'c' in dynamic
    assert arg_deps == []