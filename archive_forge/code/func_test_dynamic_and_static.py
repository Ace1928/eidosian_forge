import numpy as np
import pandas as pd
import pytest
from unittest import TestCase, SkipTest
from hvplot.util import (
def test_dynamic_and_static(self):
    import panel as pn
    from ..util import process_dynamic_args
    x = 'sepal_width'
    y = pn.widgets.Select(name='y', value='sepal_length', options=['sepal_length', 'petal_length'])
    kind = pn.widgets.Select(name='kind', value='scatter', options=['bivariate', 'scatter'])
    dynamic, arg_deps, arg_names = process_dynamic_args(x, y, kind)
    assert 'x' not in dynamic
    assert 'y' in dynamic
    assert arg_deps == []