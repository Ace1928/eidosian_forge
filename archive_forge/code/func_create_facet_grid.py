from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.figure_factory import utils
from plotly.subplots import make_subplots
import math
from numbers import Number
def create_facet_grid(df, x=None, y=None, facet_row=None, facet_col=None, color_name=None, colormap=None, color_is_cat=False, facet_row_labels=None, facet_col_labels=None, height=None, width=None, trace_type='scatter', scales='fixed', dtick_x=None, dtick_y=None, show_boxes=True, ggplot2=False, binsize=1, **kwargs):
    """
    Returns figure for facet grid; **this function is deprecated**, since
    plotly.express functions should be used instead, for example

    >>> import plotly.express as px
    >>> tips = px.data.tips()
    >>> fig = px.scatter(tips,
    ...     x='total_bill',
    ...     y='tip',
    ...     facet_row='sex',
    ...     facet_col='smoker',
    ...     color='size')


    :param (pd.DataFrame) df: the dataframe of columns for the facet grid.
    :param (str) x: the name of the dataframe column for the x axis data.
    :param (str) y: the name of the dataframe column for the y axis data.
    :param (str) facet_row: the name of the dataframe column that is used to
        facet the grid into row panels.
    :param (str) facet_col: the name of the dataframe column that is used to
        facet the grid into column panels.
    :param (str) color_name: the name of your dataframe column that will
        function as the colormap variable.
    :param (str|list|dict) colormap: the param that determines how the
        color_name column colors the data. If the dataframe contains numeric
        data, then a dictionary of colors will group the data categorically
        while a Plotly Colorscale name or a custom colorscale will treat it
        numerically. To learn more about colors and types of colormap, run
        `help(plotly.colors)`.
    :param (bool) color_is_cat: determines whether a numerical column for the
        colormap will be treated as categorical (True) or sequential (False).
            Default = False.
    :param (str|dict) facet_row_labels: set to either 'name' or a dictionary
        of all the unique values in the faceting row mapped to some text to
        show up in the label annotations. If None, labeling works like usual.
    :param (str|dict) facet_col_labels: set to either 'name' or a dictionary
        of all the values in the faceting row mapped to some text to show up
        in the label annotations. If None, labeling works like usual.
    :param (int) height: the height of the facet grid figure.
    :param (int) width: the width of the facet grid figure.
    :param (str) trace_type: decides the type of plot to appear in the
        facet grid. The options are 'scatter', 'scattergl', 'histogram',
        'bar', and 'box'.
        Default = 'scatter'.
    :param (str) scales: determines if axes have fixed ranges or not. Valid
        settings are 'fixed' (all axes fixed), 'free_x' (x axis free only),
        'free_y' (y axis free only) or 'free' (both axes free).
    :param (float) dtick_x: determines the distance between each tick on the
        x-axis. Default is None which means dtick_x is set automatically.
    :param (float) dtick_y: determines the distance between each tick on the
        y-axis. Default is None which means dtick_y is set automatically.
    :param (bool) show_boxes: draws grey boxes behind the facet titles.
    :param (bool) ggplot2: draws the facet grid in the style of `ggplot2`. See
        http://ggplot2.tidyverse.org/reference/facet_grid.html for reference.
        Default = False
    :param (int) binsize: groups all data into bins of a given length.
    :param (dict) kwargs: a dictionary of scatterplot arguments.

    Examples 1: One Way Faceting

    >>> import plotly.figure_factory as ff
    >>> import pandas as pd
    >>> mpg = pd.read_table('https://raw.githubusercontent.com/plotly/datasets/master/mpg_2017.txt')

    >>> fig = ff.create_facet_grid(
    ...     mpg,
    ...     x='displ',
    ...     y='cty',
    ...     facet_col='cyl',
    ... )
    >>> fig.show()

    Example 2: Two Way Faceting

    >>> import plotly.figure_factory as ff

    >>> import pandas as pd

    >>> mpg = pd.read_table('https://raw.githubusercontent.com/plotly/datasets/master/mpg_2017.txt')

    >>> fig = ff.create_facet_grid(
    ...     mpg,
    ...     x='displ',
    ...     y='cty',
    ...     facet_row='drv',
    ...     facet_col='cyl',
    ... )
    >>> fig.show()

    Example 3: Categorical Coloring

    >>> import plotly.figure_factory as ff
    >>> import pandas as pd
    >>> mtcars = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/mtcars.csv')
    >>> mtcars.cyl = mtcars.cyl.astype(str)
    >>> fig = ff.create_facet_grid(
    ...     mtcars,
    ...     x='mpg',
    ...     y='wt',
    ...     facet_col='cyl',
    ...     color_name='cyl',
    ...     color_is_cat=True,
    ... )
    >>> fig.show()


    """
    if not pd:
        raise ImportError("'pandas' must be installed for this figure_factory.")
    if not isinstance(df, pd.DataFrame):
        raise exceptions.PlotlyError('You must input a pandas DataFrame.')
    utils.validate_dataframe(df)
    if trace_type in ['scatter', 'scattergl']:
        if not x or not y:
            raise exceptions.PlotlyError("You need to input 'x' and 'y' if you are you are using a trace_type of 'scatter' or 'scattergl'.")
    for key in [x, y, facet_row, facet_col, color_name]:
        if key is not None:
            try:
                df[key]
            except KeyError:
                raise exceptions.PlotlyError('x, y, facet_row, facet_col and color_name must be keys in your dataframe.')
    if trace_type not in ['scatter', 'scattergl']:
        scales = 'free'
    if scales not in ['fixed', 'free_x', 'free_y', 'free']:
        raise exceptions.PlotlyError("'scales' must be set to 'fixed', 'free_x', 'free_y' and 'free'.")
    if trace_type not in VALID_TRACE_TYPES:
        raise exceptions.PlotlyError("'trace_type' must be in {}".format(VALID_TRACE_TYPES))
    if trace_type == 'histogram':
        SUBPLOT_SPACING = 0.06
    else:
        SUBPLOT_SPACING = 0.015
    if 'marker' in kwargs:
        kwargs_marker = kwargs['marker']
    else:
        kwargs_marker = {}
    marker_color = kwargs_marker.pop('color', None)
    kwargs.pop('marker', None)
    kwargs_trace = kwargs
    if 'size' not in kwargs_marker:
        if ggplot2:
            kwargs_marker['size'] = 5
        else:
            kwargs_marker['size'] = 8
    if 'opacity' not in kwargs_marker:
        if not ggplot2:
            kwargs_trace['opacity'] = 0.6
    if 'line' not in kwargs_marker:
        if not ggplot2:
            kwargs_marker['line'] = {'color': 'darkgrey', 'width': 1}
        else:
            kwargs_marker['line'] = {}
    if not ggplot2:
        if not marker_color:
            marker_color = 'rgb(31, 119, 180)'
    else:
        marker_color = 'rgb(0, 0, 0)'
    num_of_rows = 1
    num_of_cols = 1
    flipped_rows = False
    flipped_cols = False
    if facet_row:
        num_of_rows = len(df[facet_row].unique())
        flipped_rows = _is_flipped(num_of_rows)
        if isinstance(facet_row_labels, dict):
            for key in df[facet_row].unique():
                if key not in facet_row_labels.keys():
                    unique_keys = df[facet_row].unique().tolist()
                    raise exceptions.PlotlyError(CUSTOM_LABEL_ERROR.format(unique_keys))
    if facet_col:
        num_of_cols = len(df[facet_col].unique())
        flipped_cols = _is_flipped(num_of_cols)
        if isinstance(facet_col_labels, dict):
            for key in df[facet_col].unique():
                if key not in facet_col_labels.keys():
                    unique_keys = df[facet_col].unique().tolist()
                    raise exceptions.PlotlyError(CUSTOM_LABEL_ERROR.format(unique_keys))
    show_legend = False
    if color_name:
        if isinstance(df[color_name].iloc[0], str) or color_is_cat:
            show_legend = True
            if isinstance(colormap, dict):
                clrs.validate_colors_dict(colormap, 'rgb')
                for val in df[color_name].unique():
                    if val not in colormap.keys():
                        raise exceptions.PlotlyError("If using 'colormap' as a dictionary, make sure all the values of the colormap column are in the keys of your dictionary.")
            else:
                default_colors = clrs.DEFAULT_PLOTLY_COLORS
                colormap = {}
                j = 0
                for val in df[color_name].unique():
                    if j >= len(default_colors):
                        j = 0
                    colormap[val] = default_colors[j]
                    j += 1
            fig, annotations = _facet_grid_color_categorical(df, x, y, facet_row, facet_col, color_name, colormap, num_of_rows, num_of_cols, facet_row_labels, facet_col_labels, trace_type, flipped_rows, flipped_cols, show_boxes, SUBPLOT_SPACING, marker_color, kwargs_trace, kwargs_marker)
        elif isinstance(df[color_name].iloc[0], Number):
            if isinstance(colormap, dict):
                show_legend = True
                clrs.validate_colors_dict(colormap, 'rgb')
                for val in df[color_name].unique():
                    if val not in colormap.keys():
                        raise exceptions.PlotlyError("If using 'colormap' as a dictionary, make sure all the values of the colormap column are in the keys of your dictionary.")
                fig, annotations = _facet_grid_color_categorical(df, x, y, facet_row, facet_col, color_name, colormap, num_of_rows, num_of_cols, facet_row_labels, facet_col_labels, trace_type, flipped_rows, flipped_cols, show_boxes, SUBPLOT_SPACING, marker_color, kwargs_trace, kwargs_marker)
            elif isinstance(colormap, list):
                colorscale_list = colormap
                clrs.validate_colorscale(colorscale_list)
                fig, annotations = _facet_grid_color_numerical(df, x, y, facet_row, facet_col, color_name, colorscale_list, num_of_rows, num_of_cols, facet_row_labels, facet_col_labels, trace_type, flipped_rows, flipped_cols, show_boxes, SUBPLOT_SPACING, marker_color, kwargs_trace, kwargs_marker)
            elif isinstance(colormap, str):
                if colormap in clrs.PLOTLY_SCALES.keys():
                    colorscale_list = clrs.PLOTLY_SCALES[colormap]
                else:
                    raise exceptions.PlotlyError("If 'colormap' is a string, it must be the name of a Plotly Colorscale. The available colorscale names are {}".format(clrs.PLOTLY_SCALES.keys()))
                fig, annotations = _facet_grid_color_numerical(df, x, y, facet_row, facet_col, color_name, colorscale_list, num_of_rows, num_of_cols, facet_row_labels, facet_col_labels, trace_type, flipped_rows, flipped_cols, show_boxes, SUBPLOT_SPACING, marker_color, kwargs_trace, kwargs_marker)
            else:
                colorscale_list = clrs.PLOTLY_SCALES['Reds']
                fig, annotations = _facet_grid_color_numerical(df, x, y, facet_row, facet_col, color_name, colorscale_list, num_of_rows, num_of_cols, facet_row_labels, facet_col_labels, trace_type, flipped_rows, flipped_cols, show_boxes, SUBPLOT_SPACING, marker_color, kwargs_trace, kwargs_marker)
    else:
        fig, annotations = _facet_grid(df, x, y, facet_row, facet_col, num_of_rows, num_of_cols, facet_row_labels, facet_col_labels, trace_type, flipped_rows, flipped_cols, show_boxes, SUBPLOT_SPACING, marker_color, kwargs_trace, kwargs_marker)
    if not height:
        height = max(600, 100 * num_of_rows)
    if not width:
        width = max(600, 100 * num_of_cols)
    fig['layout'].update(height=height, width=width, title='', paper_bgcolor='rgb(251, 251, 251)')
    if ggplot2:
        fig['layout'].update(plot_bgcolor=PLOT_BGCOLOR, paper_bgcolor='rgb(255, 255, 255)', hovermode='closest')
    x_title_annot = _axis_title_annotation(x, 'x')
    y_title_annot = _axis_title_annotation(y, 'y')
    annotations.append(x_title_annot)
    annotations.append(y_title_annot)
    fig['layout']['showlegend'] = show_legend
    fig['layout']['legend']['bgcolor'] = LEGEND_COLOR
    fig['layout']['legend']['borderwidth'] = LEGEND_BORDER_WIDTH
    fig['layout']['legend']['x'] = 1.05
    fig['layout']['legend']['y'] = 1
    fig['layout']['legend']['yanchor'] = 'top'
    if show_legend:
        fig['layout']['showlegend'] = show_legend
        if ggplot2:
            if color_name:
                legend_annot = _legend_annotation(color_name)
                annotations.append(legend_annot)
            fig['layout']['margin']['r'] = 150
    fig['layout']['annotations'] = annotations
    if show_boxes and ggplot2:
        _add_shapes_to_fig(fig, ANNOT_RECT_COLOR, flipped_rows, flipped_cols)
    axis_labels = {'x': [], 'y': []}
    for key in fig['layout']:
        if 'xaxis' in key:
            axis_labels['x'].append(key)
        elif 'yaxis' in key:
            axis_labels['y'].append(key)
    string_number_in_data = False
    for var in [v for v in [x, y] if v]:
        if isinstance(df[var].tolist()[0], str):
            for item in df[var]:
                try:
                    int(item)
                    string_number_in_data = True
                except ValueError:
                    pass
    if string_number_in_data:
        for x_y in axis_labels.keys():
            for axis_name in axis_labels[x_y]:
                fig['layout'][axis_name]['type'] = 'category'
    if scales == 'fixed':
        fixed_axes = ['x', 'y']
    elif scales == 'free_x':
        fixed_axes = ['y']
    elif scales == 'free_y':
        fixed_axes = ['x']
    elif scales == 'free':
        fixed_axes = []
    for x_y in fixed_axes:
        min_ranges = []
        max_ranges = []
        for trace in fig['data']:
            if trace[x_y] is not None and len(trace[x_y]) > 0:
                min_ranges.append(min(trace[x_y]))
                max_ranges.append(max(trace[x_y]))
        while None in min_ranges:
            min_ranges.remove(None)
        while None in max_ranges:
            max_ranges.remove(None)
        min_range = min(min_ranges)
        max_range = max(max_ranges)
        range_are_numbers = isinstance(min_range, Number) and isinstance(max_range, Number)
        if range_are_numbers:
            min_range = math.floor(min_range)
            max_range = math.ceil(max_range)
            min_range -= 0.05 * (max_range - min_range)
            max_range += 0.05 * (max_range - min_range)
            if x_y == 'x':
                if dtick_x:
                    dtick = dtick_x
                else:
                    dtick = math.floor((max_range - min_range) / MAX_TICKS_PER_AXIS)
            elif x_y == 'y':
                if dtick_y:
                    dtick = dtick_y
                else:
                    dtick = math.floor((max_range - min_range) / MAX_TICKS_PER_AXIS)
        else:
            dtick = 1
        for axis_title in axis_labels[x_y]:
            fig['layout'][axis_title]['dtick'] = dtick
            fig['layout'][axis_title]['ticklen'] = 0
            fig['layout'][axis_title]['zeroline'] = False
            if ggplot2:
                fig['layout'][axis_title]['tickwidth'] = 1
                fig['layout'][axis_title]['ticklen'] = 4
                fig['layout'][axis_title]['gridwidth'] = GRID_WIDTH
                fig['layout'][axis_title]['gridcolor'] = GRID_COLOR
                fig['layout'][axis_title]['gridwidth'] = 2
                fig['layout'][axis_title]['tickfont'] = {'color': TICK_COLOR, 'size': 10}
        if x_y in fixed_axes:
            for key in fig['layout']:
                if '{}axis'.format(x_y) in key and range_are_numbers:
                    fig['layout'][key]['range'] = [min_range, max_range]
    return fig