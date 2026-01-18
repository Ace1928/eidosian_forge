import pytest
import numpy as np
from traitlets import HasTraits, TraitError, observe
from ipywidgets import Widget
from ..ndarray.traits import shape_constraints
from ..ndarray.union import DataUnion, get_union_array
from ..ndarray.widgets import NDArrayWidget, NDArraySource
@observe('bar')
def on_bar_change(self, change):
    ns['counter'] += 1