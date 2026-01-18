import warnings
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.category as cat
from matplotlib.testing.decorators import check_figures_equal
def axis_test(axis, labels):
    ticks = list(range(len(labels)))
    np.testing.assert_array_equal(axis.get_majorticklocs(), ticks)
    graph_labels = [axis.major.formatter(i, i) for i in ticks]
    assert graph_labels == [cat.StrCategoryFormatter._text(l) for l in labels]
    assert list(axis.units._mapping.keys()) == [l for l in labels]
    assert list(axis.units._mapping.values()) == ticks