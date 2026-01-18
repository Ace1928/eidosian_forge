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
def plot_features_selection_loss_graph(title, entities_name, entities_name_in_fields, eliminated_entities_indices, eliminated_entities_names, loss_graph, cost_graph=None):
    warn_msg = 'To draw plots you should install plotly.'
    try:
        import plotly.graph_objs as go
    except ImportError as e:
        warnings.warn(warn_msg)
        raise ImportError(str(e))
    indices_present = any(eliminated_entities_indices)
    names_present = any(eliminated_entities_names)
    names_or_indices = eliminated_entities_names if names_present else list(map(str, eliminated_entities_indices))
    loss_values = loss_graph['loss_values']
    removed_entities_cnt = loss_graph['removed_' + entities_name_in_fields + '_count']
    main_indices = loss_graph['main_indices']
    fig = go.Figure()
    fig['layout']['title'] = go.layout.Title(text=title)
    loss_graph_color = 'rgb(51,160,44)'
    fig.add_trace(go.Scatter(x=removed_entities_cnt, y=loss_values, line=go.scatter.Line(color=loss_graph_color), mode='lines+markers', text=[''] + names_or_indices, name=''))
    if len(main_indices) > 0:
        fig.add_trace(go.Scatter(x=[removed_entities_cnt[idx] for idx in main_indices], y=[loss_values[idx] for idx in main_indices], mode='markers', marker=go.scatter.Marker(size=10, symbol='square'), text=[names_or_indices[idx - 1] if idx > 0 else '' for idx in main_indices], name=''))
    if indices_present:
        fig.add_trace(go.Scatter(x=removed_entities_cnt, y=loss_values, mode='text', text=[''] + list(map(str, eliminated_entities_indices)), textposition='bottom center', textfont=dict(family='sans serif', size=18, color=loss_graph_color), name='', visible=False))
    if names_present:
        fig.add_trace(go.Scatter(x=removed_entities_cnt, y=loss_values, mode='text', text=[''] + eliminated_entities_names, textfont=dict(family='sans serif', size=18, color=loss_graph_color), textposition='bottom center', name='', visible=False))
    cost_graph_color = 'rgb(160,44,44)'
    if cost_graph is not None:
        fig.add_trace(go.Scatter(x=removed_entities_cnt, y=cost_graph['loss_values'], line=go.scatter.Line(color=cost_graph_color), mode='lines+markers', text=[''] + eliminated_entities_names, name='', yaxis='y2'))
        fig.add_trace(go.Scatter(x=removed_entities_cnt, y=cost_graph['loss_values'], mode='text', text=[''] + eliminated_entities_names, textfont=dict(family='sans serif', size=18, color=cost_graph_color), textposition='bottom center', name='', yaxis='y2', visible=False))
    axis_options = dict(gridcolor='rgb(255,255,255)', showgrid=True, showline=False, showticklabels=True, tickcolor='rgb(127,127,127)', ticks='outside', zeroline=False)
    fig.update_layout(xaxis=dict(title='number of removed ' + entities_name, **axis_options), yaxis=dict(title='loss value', titlefont=dict(color=loss_graph_color), tickfont=dict(color=loss_graph_color), **axis_options))
    if cost_graph is not None:
        fig.update_layout(yaxis2=dict(title='cost value', side='right', anchor='x', overlaying='y', titlefont=dict(color=cost_graph_color), tickfont=dict(color=cost_graph_color), **axis_options))
    buttons = []

    def get_visible_arg(show_indices, show_names):
        visible_arg = [True]
        if len(main_indices) > 0:
            visible_arg.append(True)
        if indices_present:
            visible_arg.append(show_indices)
        if names_present:
            visible_arg.append(show_names)
        if cost_graph is not None:
            visible_arg.append(True)
            visible_arg.append(show_names)
        return visible_arg
    buttons.append(dict(label='Hide ' + entities_name, method='update', args=[{'visible': get_visible_arg(show_indices=False, show_names=False)}]))
    if indices_present:
        buttons.append(dict(label='Show indices', method='update', args=[{'visible': get_visible_arg(show_indices=True, show_names=False)}]))
    if names_present:
        buttons.append(dict(label='Show names', method='update', args=[{'visible': get_visible_arg(show_indices=False, show_names=True)}]))
    fig.update_layout(updatemenus=[dict(active=0, buttons=buttons, pad={'r': 10, 't': 10}, showactive=True, x=-0.25, xanchor='left', y=1.03, yanchor='top')])
    fig.update_layout(showlegend=False)
    return fig