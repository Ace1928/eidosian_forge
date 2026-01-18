import plotly.graph_objs as go
import plotly.io as pio
from collections import namedtuple, OrderedDict
from ._special_inputs import IdentityMap, Constant, Range
from .trendline_functions import ols, lowess, rolling, expanding, ewm
from _plotly_utils.basevalidators import ColorscaleValidator
from plotly.colors import qualitative, sequential
import math
from packaging import version
import pandas as pd
import numpy as np
from plotly._subplots import (
def infer_config(args, constructor, trace_patch, layout_patch):
    attrs = [k for k in direct_attrables + array_attrables if k in args]
    grouped_attrs = []
    sizeref = 0
    if 'size' in args and args['size']:
        sizeref = args['data_frame'][args['size']].max() / args['size_max'] ** 2
    if 'color' in args:
        if 'color_continuous_scale' in args:
            if 'color_discrete_sequence' not in args:
                attrs.append('color')
            elif args['color'] and _is_continuous(args['data_frame'], args['color']):
                attrs.append('color')
                args['color_is_continuous'] = True
            elif constructor in [go.Sunburst, go.Treemap, go.Icicle]:
                attrs.append('color')
                args['color_is_continuous'] = False
            else:
                grouped_attrs.append('marker.color')
        elif 'line_group' in args or constructor == go.Histogram2dContour:
            grouped_attrs.append('line.color')
        elif constructor in [go.Pie, go.Funnelarea]:
            attrs.append('color')
            if args['color']:
                if args['hover_data'] is None:
                    args['hover_data'] = []
                args['hover_data'].append(args['color'])
        else:
            grouped_attrs.append('marker.color')
        show_colorbar = bool('color' in attrs and args['color'] and (constructor not in [go.Pie, go.Funnelarea]) and (constructor not in [go.Treemap, go.Sunburst, go.Icicle] or args.get('color_is_continuous')))
    else:
        show_colorbar = False
    if 'line_dash' in args:
        grouped_attrs.append('line.dash')
    if 'symbol' in args:
        grouped_attrs.append('marker.symbol')
    if 'pattern_shape' in args:
        if constructor in [go.Scatter]:
            grouped_attrs.append('fillpattern.shape')
        else:
            grouped_attrs.append('marker.pattern.shape')
    if 'orientation' in args:
        has_x = args['x'] is not None
        has_y = args['y'] is not None
        if args['orientation'] is None:
            if constructor in [go.Histogram, go.Scatter]:
                if has_y and (not has_x):
                    args['orientation'] = 'h'
            elif constructor in [go.Violin, go.Box, go.Bar, go.Funnel]:
                if has_x and (not has_y):
                    args['orientation'] = 'h'
        if args['orientation'] is None and has_x and has_y:
            x_is_continuous = _is_continuous(args['data_frame'], args['x'])
            y_is_continuous = _is_continuous(args['data_frame'], args['y'])
            if x_is_continuous and (not y_is_continuous):
                args['orientation'] = 'h'
            if y_is_continuous and (not x_is_continuous):
                args['orientation'] = 'v'
        if args['orientation'] is None:
            args['orientation'] = 'v'
        if constructor == go.Histogram:
            if has_x and has_y and (args['histfunc'] is None):
                args['histfunc'] = trace_patch['histfunc'] = 'sum'
            orientation = args['orientation']
            nbins = args['nbins']
            trace_patch['nbinsx'] = nbins if orientation == 'v' else None
            trace_patch['nbinsy'] = None if orientation == 'v' else nbins
            trace_patch['bingroup'] = 'x' if orientation == 'v' else 'y'
        trace_patch['orientation'] = args['orientation']
        if constructor in [go.Violin, go.Box]:
            mode = 'boxmode' if constructor == go.Box else 'violinmode'
            if layout_patch[mode] is None and args['color'] is not None:
                if args['y'] == args['color'] and args['orientation'] == 'h':
                    layout_patch[mode] = 'overlay'
                elif args['x'] == args['color'] and args['orientation'] == 'v':
                    layout_patch[mode] = 'overlay'
            if layout_patch[mode] is None:
                layout_patch[mode] = 'group'
    if constructor == go.Histogram2d and args['z'] is not None and (args['histfunc'] is None):
        args['histfunc'] = trace_patch['histfunc'] = 'sum'
    if args.get('text_auto', False) is not False:
        if constructor in [go.Histogram2d, go.Histogram2dContour]:
            letter = 'z'
        elif constructor == go.Bar:
            letter = 'y' if args['orientation'] == 'v' else 'x'
        else:
            letter = 'value'
        if args['text_auto'] is True:
            trace_patch['texttemplate'] = '%{' + letter + '}'
        else:
            trace_patch['texttemplate'] = '%{' + letter + ':' + args['text_auto'] + '}'
    if constructor in [go.Histogram2d, go.Densitymapbox]:
        show_colorbar = True
        trace_patch['coloraxis'] = 'coloraxis1'
    if 'opacity' in args:
        if args['opacity'] is None:
            if 'barmode' in args and args['barmode'] == 'overlay':
                trace_patch['marker'] = dict(opacity=0.5)
        elif constructor in [go.Densitymapbox, go.Pie, go.Funnel, go.Funnelarea]:
            trace_patch['opacity'] = args['opacity']
        else:
            trace_patch['marker'] = dict(opacity=args['opacity'])
    if 'line_group' in args or 'line_dash' in args:
        modes = set()
        if args.get('lines', True):
            modes.add('lines')
        if args.get('text') or args.get('symbol') or args.get('markers'):
            modes.add('markers')
        if args.get('text'):
            modes.add('text')
        if len(modes) == 0:
            modes.add('lines')
        trace_patch['mode'] = '+'.join(sorted(modes))
    elif constructor != go.Splom and ('symbol' in args or constructor == go.Scattermapbox):
        trace_patch['mode'] = 'markers' + ('+text' if args['text'] else '')
    if 'line_shape' in args:
        trace_patch['line'] = dict(shape=args['line_shape'])
    elif 'ecdfmode' in args:
        trace_patch['line'] = dict(shape='vh' if args['ecdfmode'] == 'reversed' else 'hv')
    if 'geojson' in args:
        trace_patch['featureidkey'] = args['featureidkey']
        trace_patch['geojson'] = args['geojson'] if not hasattr(args['geojson'], '__geo_interface__') else args['geojson'].__geo_interface__
    if 'marginal' in args:
        position = 'marginal_x' if args['orientation'] == 'v' else 'marginal_y'
        other_position = 'marginal_x' if args['orientation'] == 'h' else 'marginal_y'
        args[position] = args['marginal']
        args[other_position] = None
    if len(args['data_frame']) == 0:
        args['facet_row'] = args['facet_col'] = None
    if args.get('facet_col') is not None and args.get('marginal_y') is not None:
        args['marginal_y'] = None
    if args.get('facet_row') is not None and args.get('marginal_x') is not None:
        args['marginal_x'] = None
    if args.get('marginal_x') is not None or args.get('marginal_y') is not None or args.get('facet_row') is not None:
        args['facet_col_wrap'] = 0
    if 'trendline' in args and args['trendline'] is not None:
        if args['trendline'] not in trendline_functions:
            raise ValueError("Value '%s' for `trendline` must be one of %s" % (args['trendline'], trendline_functions.keys()))
    if 'trendline_options' in args and args['trendline_options'] is None:
        args['trendline_options'] = dict()
    if 'ecdfnorm' in args:
        if args.get('ecdfnorm', None) not in [None, 'percent', 'probability']:
            raise ValueError("`ecdfnorm` must be one of None, 'percent' or 'probability'. " + "'%s' was provided." % args['ecdfnorm'])
        args['histnorm'] = args['ecdfnorm']
    for k in group_attrables:
        if k in args:
            grouped_attrs.append(k)
    grouped_mappings = [make_mapping(args, a) for a in grouped_attrs]
    trace_specs = make_trace_spec(args, constructor, attrs, trace_patch)
    return (trace_specs, grouped_mappings, sizeref, show_colorbar)