from datetime import (
import gc
import itertools
import re
import string
import weakref
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_df_gridspec_patterns_boxed(self):
    from matplotlib import gridspec
    import matplotlib.pyplot as plt
    ts = Series(np.random.default_rng(2).standard_normal(10), index=date_range('1/1/2000', periods=10))

    def _get_boxed_grid():
        gs = gridspec.GridSpec(3, 3)
        fig = plt.figure()
        ax1 = fig.add_subplot(gs[:2, :2])
        ax2 = fig.add_subplot(gs[:2, 2])
        ax3 = fig.add_subplot(gs[2, :2])
        ax4 = fig.add_subplot(gs[2, 2])
        return (ax1, ax2, ax3, ax4)
    axes = _get_boxed_grid()
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), index=ts.index, columns=list('ABCD'))
    axes = df.plot(subplots=True, ax=axes)
    for ax in axes:
        assert len(ax.lines) == 1
        _check_visible(ax.get_yticklabels(), visible=True)
        _check_visible(ax.get_xticklabels(), visible=True)
        _check_visible(ax.get_xticklabels(minor=True), visible=True)
    plt.close('all')
    axes = _get_boxed_grid()
    with tm.assert_produces_warning(UserWarning):
        axes = df.plot(subplots=True, ax=axes, sharex=True, sharey=True)
    for ax in axes:
        assert len(ax.lines) == 1
    for ax in [axes[0], axes[2]]:
        _check_visible(ax.get_yticklabels(), visible=True)
    for ax in [axes[1], axes[3]]:
        _check_visible(ax.get_yticklabels(), visible=False)
    for ax in [axes[0], axes[1]]:
        _check_visible(ax.get_xticklabels(), visible=False)
        _check_visible(ax.get_xticklabels(minor=True), visible=False)
    for ax in [axes[2], axes[3]]:
        _check_visible(ax.get_xticklabels(), visible=True)
        _check_visible(ax.get_xticklabels(minor=True), visible=True)
    plt.close('all')