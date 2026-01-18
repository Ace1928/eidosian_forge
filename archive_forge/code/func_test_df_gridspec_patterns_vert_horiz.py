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
def test_df_gridspec_patterns_vert_horiz(self):
    from matplotlib import gridspec
    import matplotlib.pyplot as plt
    ts = Series(np.random.default_rng(2).standard_normal(10), index=date_range('1/1/2000', periods=10))
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), index=ts.index, columns=list('AB'))

    def _get_vertical_grid():
        gs = gridspec.GridSpec(3, 1)
        fig = plt.figure()
        ax1 = fig.add_subplot(gs[:2, :])
        ax2 = fig.add_subplot(gs[2, :])
        return (ax1, ax2)

    def _get_horizontal_grid():
        gs = gridspec.GridSpec(1, 3)
        fig = plt.figure()
        ax1 = fig.add_subplot(gs[:, :2])
        ax2 = fig.add_subplot(gs[:, 2])
        return (ax1, ax2)
    for ax1, ax2 in [_get_vertical_grid(), _get_horizontal_grid()]:
        ax1 = ts.plot(ax=ax1)
        assert len(ax1.lines) == 1
        ax2 = df.plot(ax=ax2)
        assert len(ax2.lines) == 2
        for ax in [ax1, ax2]:
            _check_visible(ax.get_yticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(minor=True), visible=True)
        plt.close('all')
    for ax1, ax2 in [_get_vertical_grid(), _get_horizontal_grid()]:
        axes = df.plot(subplots=True, ax=[ax1, ax2])
        assert len(ax1.lines) == 1
        assert len(ax2.lines) == 1
        for ax in axes:
            _check_visible(ax.get_yticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(minor=True), visible=True)
        plt.close('all')
    ax1, ax2 = _get_vertical_grid()
    with tm.assert_produces_warning(UserWarning):
        axes = df.plot(subplots=True, ax=[ax1, ax2], sharex=True, sharey=True)
    assert len(axes[0].lines) == 1
    assert len(axes[1].lines) == 1
    for ax in [ax1, ax2]:
        _check_visible(ax.get_yticklabels(), visible=True)
    _check_visible(axes[0].get_xticklabels(), visible=False)
    _check_visible(axes[0].get_xticklabels(minor=True), visible=False)
    _check_visible(axes[1].get_xticklabels(), visible=True)
    _check_visible(axes[1].get_xticklabels(minor=True), visible=True)
    plt.close('all')
    ax1, ax2 = _get_horizontal_grid()
    with tm.assert_produces_warning(UserWarning):
        axes = df.plot(subplots=True, ax=[ax1, ax2], sharex=True, sharey=True)
    assert len(axes[0].lines) == 1
    assert len(axes[1].lines) == 1
    _check_visible(axes[0].get_yticklabels(), visible=True)
    _check_visible(axes[1].get_yticklabels(), visible=False)
    for ax in [ax1, ax2]:
        _check_visible(ax.get_xticklabels(), visible=True)
        _check_visible(ax.get_xticklabels(minor=True), visible=True)
    plt.close('all')