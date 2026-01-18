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
def make_figure(args, constructor, trace_patch=None, layout_patch=None):
    trace_patch = trace_patch or {}
    layout_patch = layout_patch or {}
    apply_default_cascade(args)
    args = build_dataframe(args, constructor)
    if constructor in [go.Treemap, go.Sunburst, go.Icicle] and args['path'] is not None:
        args = process_dataframe_hierarchy(args)
    if constructor in [go.Pie]:
        args, trace_patch = process_dataframe_pie(args, trace_patch)
    if constructor == 'timeline':
        constructor = go.Bar
        args = process_dataframe_timeline(args)
    trace_specs, grouped_mappings, sizeref, show_colorbar = infer_config(args, constructor, trace_patch, layout_patch)
    grouper = [x.grouper or one_group for x in grouped_mappings] or [one_group]
    groups, orders = get_groups_and_orders(args, grouper)
    col_labels = []
    row_labels = []
    nrows = ncols = 1
    for m in grouped_mappings:
        if m.grouper not in orders:
            m.val_map[''] = m.sequence[0]
        else:
            sorted_values = orders[m.grouper]
            if m.facet == 'col':
                prefix = get_label(args, args['facet_col']) + '='
                col_labels = [prefix + str(s) for s in sorted_values]
                ncols = len(col_labels)
            if m.facet == 'row':
                prefix = get_label(args, args['facet_row']) + '='
                row_labels = [prefix + str(s) for s in sorted_values]
                nrows = len(row_labels)
            for val in sorted_values:
                if val not in m.val_map:
                    m.val_map[val] = m.sequence[len(m.val_map) % len(m.sequence)]
    subplot_type = _subplot_type_for_trace_type(constructor().type)
    trace_names_by_frame = {}
    frames = OrderedDict()
    trendline_rows = []
    trace_name_labels = None
    facet_col_wrap = args.get('facet_col_wrap', 0)
    for group_name, group in groups.items():
        mapping_labels = OrderedDict()
        trace_name_labels = OrderedDict()
        frame_name = ''
        for col, val, m in zip(grouper, group_name, grouped_mappings):
            if col != one_group:
                key = get_label(args, col)
                if not isinstance(m.val_map, IdentityMap):
                    mapping_labels[key] = str(val)
                    if m.show_in_trace_name:
                        trace_name_labels[key] = str(val)
                if m.variable == 'animation_frame':
                    frame_name = val
        trace_name = ', '.join(trace_name_labels.values())
        if frame_name not in trace_names_by_frame:
            trace_names_by_frame[frame_name] = set()
        trace_names = trace_names_by_frame[frame_name]
        for trace_spec in trace_specs:
            trace = trace_spec.constructor(name=trace_name)
            if trace_spec.constructor not in [go.Parcats, go.Parcoords, go.Choropleth, go.Choroplethmapbox, go.Densitymapbox, go.Histogram2d, go.Sunburst, go.Treemap, go.Icicle]:
                trace.update(legendgroup=trace_name, showlegend=trace_name != '' and trace_name not in trace_names)
            if trace_spec.constructor in [go.Bar, go.Violin, go.Box, go.Histogram]:
                trace.update(alignmentgroup=True, offsetgroup=trace_name)
            trace_names.add(trace_name)
            trace._subplot_row = 1
            trace._subplot_col = 1
            for i, m in enumerate(grouped_mappings):
                val = group_name[i]
                try:
                    m.updater(trace, m.val_map[val])
                except ValueError:
                    if trace_spec != trace_specs[0] and (trace_spec.constructor in [go.Violin, go.Box] and m.variable in ['symbol', 'pattern', 'dash']) or (trace_spec.constructor in [go.Histogram] and m.variable in ['symbol', 'dash']):
                        pass
                    elif trace_spec != trace_specs[0] and trace_spec.constructor in [go.Histogram] and (m.variable == 'color'):
                        trace.update(marker=dict(color=m.val_map[val]))
                    elif trace_spec.constructor in [go.Choropleth, go.Choroplethmapbox] and m.variable == 'color':
                        trace.update(z=[1] * len(group), colorscale=[m.val_map[val]] * 2, showscale=False, showlegend=True)
                    else:
                        raise
                if m.facet == 'row':
                    row = m.val_map[val]
                elif args.get('marginal_x') is not None and trace_spec.marginal != 'x':
                    row = 2
                else:
                    row = 1
                if m.facet == 'col':
                    col = m.val_map[val]
                    if facet_col_wrap:
                        row = 1 + (col - 1) // facet_col_wrap
                        col = 1 + (col - 1) % facet_col_wrap
                elif trace_spec.marginal == 'y':
                    col = 2
                else:
                    col = 1
                if row > 1:
                    trace._subplot_row = row
                if col > 1:
                    trace._subplot_col = col
            if trace_specs[0].constructor == go.Histogram2dContour and trace_spec.constructor == go.Box and trace.line.color:
                trace.update(marker=dict(color=trace.line.color))
            if 'ecdfmode' in args:
                base = args['x'] if args['orientation'] == 'v' else args['y']
                var = args['x'] if args['orientation'] == 'h' else args['y']
                ascending = args.get('ecdfmode', 'standard') != 'reversed'
                group = group.sort_values(by=base, ascending=ascending)
                group_sum = group[var].sum()
                group[var] = group[var].cumsum()
                if not ascending:
                    group = group.sort_values(by=base, ascending=True)
                if args.get('ecdfmode', 'standard') == 'complementary':
                    group[var] = group_sum - group[var]
                if args['ecdfnorm'] == 'probability':
                    group[var] = group[var] / group_sum
                elif args['ecdfnorm'] == 'percent':
                    group[var] = 100.0 * group[var] / group_sum
            patch, fit_results = make_trace_kwargs(args, trace_spec, group, mapping_labels.copy(), sizeref)
            trace.update(patch)
            if fit_results is not None:
                trendline_rows.append(mapping_labels.copy())
                trendline_rows[-1]['px_fit_results'] = fit_results
            if frame_name not in frames:
                frames[frame_name] = dict(data=[], name=frame_name)
            frames[frame_name]['data'].append(trace)
    frame_list = [f for f in frames.values()]
    if len(frame_list) > 1:
        frame_list = sorted(frame_list, key=lambda f: orders[args['animation_frame']].index(f['name']))
    if show_colorbar:
        colorvar = 'z' if constructor in [go.Histogram2d, go.Densitymapbox] else 'color'
        range_color = args['range_color'] or [None, None]
        colorscale_validator = ColorscaleValidator('colorscale', 'make_figure')
        layout_patch['coloraxis1'] = dict(colorscale=colorscale_validator.validate_coerce(args['color_continuous_scale']), cmid=args['color_continuous_midpoint'], cmin=range_color[0], cmax=range_color[1], colorbar=dict(title_text=get_decorated_label(args, args[colorvar], colorvar)))
    for v in ['height', 'width']:
        if args[v]:
            layout_patch[v] = args[v]
    layout_patch['legend'] = dict(tracegroupgap=0)
    if trace_name_labels:
        layout_patch['legend']['title_text'] = ', '.join(trace_name_labels)
    if args['title']:
        layout_patch['title_text'] = args['title']
    elif args['template'].layout.margin.t is None:
        layout_patch['margin'] = {'t': 60}
    if 'size' in args and args['size'] and (args['template'].layout.legend.itemsizing is None):
        layout_patch['legend']['itemsizing'] = 'constant'
    if facet_col_wrap:
        nrows = math.ceil(ncols / facet_col_wrap)
        ncols = min(ncols, facet_col_wrap)
    if args.get('marginal_x') is not None:
        nrows += 1
    if args.get('marginal_y') is not None:
        ncols += 1
    fig = init_figure(args, subplot_type, frame_list, nrows, ncols, col_labels, row_labels)
    for frame in frame_list:
        for trace in frame['data']:
            if isinstance(trace, go.Splom):
                continue
            _set_trace_grid_reference(trace, fig.layout, fig._grid_ref, nrows - trace._subplot_row + 1, trace._subplot_col)
    fig.add_traces(frame_list[0]['data'] if len(frame_list) > 0 else [])
    fig.update_layout(layout_patch)
    if 'template' in args and args['template'] is not None:
        fig.update_layout(template=args['template'], overwrite=True)
    for f in frame_list:
        f['name'] = str(f['name'])
    fig.frames = frame_list if len(frames) > 1 else []
    if args.get('trendline') and args.get('trendline_scope', 'trace') == 'overall':
        trendline_spec = make_trendline_spec(args, constructor)
        trendline_trace = trendline_spec.constructor(name='Overall Trendline', legendgroup='Overall Trendline', showlegend=False)
        if 'line' not in trendline_spec.trace_patch:
            for m in grouped_mappings:
                if m.variable == 'color':
                    next_color = m.sequence[len(m.val_map) % len(m.sequence)]
                    trendline_spec.trace_patch['line'] = dict(color=next_color)
        patch, fit_results = make_trace_kwargs(args, trendline_spec, args['data_frame'], {}, sizeref)
        trendline_trace.update(patch)
        fig.add_trace(trendline_trace, row='all', col='all', exclude_empty_subplots=True)
        fig.update_traces(selector=-1, showlegend=True)
        if fit_results is not None:
            trendline_rows.append(dict(px_fit_results=fit_results))
    fig._px_trendlines = pd.DataFrame(trendline_rows)
    configure_axes(args, constructor, fig, orders)
    configure_animation_controls(args, constructor, fig)
    return fig