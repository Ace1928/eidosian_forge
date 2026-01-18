from contextlib import contextmanager  # noqa E402
from copy import deepcopy
import logging
import sys
import os
from collections import OrderedDict, defaultdict
from six import iteritems, string_types, integer_types
import warnings
import numpy as np
import ctypes
import platform
import tempfile
import shutil
import json
from enum import Enum
from operator import itemgetter
import threading
import scipy.sparse
from .plot_helpers import save_plot_file, try_plot_offline, OfflineMetricVisualizer
from . import _catboost
from .metrics import BuiltinMetric
def plot_partial_dependence(self, data, features, plot=True, plot_file=None, thread_count=-1):
    """
        To use this function, you should install plotly.
        data: numpy.ndarray or pandas.DataFrame or catboost.Pool
        features: int, str, list<int>, tuple<int>, list<string>, tuple<string>
            Float features to calculate partial dependence for. Number of features should be 1 or 2.
        plot: bool
            Plot predictions.
        plot_file: str
            Output file for plot predictions.
        thread_count: int
            Number of threads to use. If -1 use maximum available number of threads.
        Returns
        -------
            If number of features is one - 1d numpy array and figure with line plot.
            If number of features is two - 2d numpy array and figure with 2d heatmap.
        """
    try:
        import plotly.graph_objs as go
    except ImportError as e:
        warnings.warn('To draw plots you should install plotly.')
        raise ImportError(str(e))

    def getFeatureIdx(feature):
        if not isinstance(feature, int):
            if self.feature_names_ is None or feature not in self.feature_names_:
                raise CatBoostError('No feature named "{}" in model'.format(feature))
            feature_idx = self.feature_names_.index(feature)
        else:
            feature_idx = feature
        assert feature_idx in self._get_borders(), 'only float features indexes are supported'
        assert len(self._get_borders()[feature_idx]) > 0, 'feature with idx {} is not used in model'.format(feature_idx)
        return feature_idx

    def getFeatureIndices(features):
        if isinstance(features, list) or isinstance(features, tuple):
            features_idxs = [getFeatureIdx(feature) for feature in features]
        elif isinstance(features, int) or isinstance(features, str):
            features_idxs = [getFeatureIdx(features)]
        else:
            raise CatBoostError("Unsupported type for argument 'features'. Must be one of: int, string, list<string>, list<int>, tuple<int>, tuple<string>")
        return features_idxs

    def getAxisParams(borders, feature_name=None):
        return {'title': 'Bins' if feature_name is None else "Bins of feature '{}'".format(feature_name), 'tickmode': 'array', 'tickvals': list(range(len(borders) + 1)), 'ticktext': ['(-inf, {:.4f}]'.format(borders[0])] + ['({:.4f}, {:.4f}]'.format(val_1, val_2) for val_1, val_2 in zip(borders[:-1], borders[1:])] + ['({:.4f}, +inf)'.format(borders[-1])], 'showticklabels': False}

    def plot2d(feature_names, borders, predictions):
        xaxis = go.layout.XAxis(**getAxisParams(borders[1], feature_name=feature_names[1]))
        yaxis = go.layout.YAxis(**getAxisParams(borders[0], feature_name=feature_names[0]))
        layout = go.Layout(title='Partial dependence plot for features {}'.format("'{}'".format("', '".join(map(str, feature_names)))), yaxis=yaxis, xaxis=xaxis)
        fig = go.Figure(data=go.Heatmap(z=predictions), layout=layout)
        return fig

    def plot1d(feature, borders, predictions):
        xaxis = go.layout.XAxis(**getAxisParams(borders))
        yaxis = {'title': 'Mean Prediction', 'side': 'left'}
        layout = go.Layout(title="Partial dependence plot for feature '{}'".format(feature), yaxis=yaxis, xaxis=xaxis)
        fig = go.Figure(data=go.Scatter(y=predictions, mode='lines+markers'), layout=layout)
        return fig
    features_idx = getFeatureIndices(features)
    borders = [self._get_borders()[idx] for idx in features_idx]
    if len(features_idx) not in [1, 2]:
        raise CatBoostError("Number of 'features' should be 1 or 2, got {}".format(len(features_idx)))
    is_2d_plot = len(features_idx) == 2
    data, _ = self._process_predict_input_data(data, 'plot_partial_dependence', thread_count=thread_count)
    all_predictions = np.array(self._object._calc_partial_dependence(data, features_idx, thread_count))
    if is_2d_plot:
        all_predictions = all_predictions.reshape([len(x) + 1 for x in borders])
        fig = plot2d(features_idx, borders, all_predictions)
    else:
        fig = plot1d(features_idx[0], borders[0], all_predictions)
    if plot:
        try_plot_offline(fig)
    if plot_file:
        save_plot_file(plot_file, "Partial dependence plot for features '{}'".format(features), fig)
    return (all_predictions, fig)