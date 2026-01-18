import gc
import numpy as np
import pytest
from pandas import (
import matplotlib as mpl
from pandas.io.formats.style import Styler
@pytest.fixture(autouse=True)
def mpl_cleanup():
    mpl = pytest.importorskip('matplotlib')
    mpl_units = pytest.importorskip('matplotlib.units')
    plt = pytest.importorskip('matplotlib.pyplot')
    orig_units_registry = mpl_units.registry.copy()
    with mpl.rc_context():
        mpl.use('template')
        yield
    mpl_units.registry.clear()
    mpl_units.registry.update(orig_units_registry)
    plt.close('all')
    gc.collect(1)