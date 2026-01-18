from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
from plotly.subplots import make_subplots
def scatterplot_dict(dataframe, headers, diag, size, height, width, title, index, index_vals, endpts, colormap, colormap_type, **kwargs):
    """
    Refer to FigureFactory.create_scatterplotmatrix() for docstring

    Returns fig for scatterplotmatrix with both index and colormap picked.
    Used if colormap is a dictionary with index values as keys pointing to
    colors. Forces colormap_type to behave categorically because it would
    not make sense colors are assigned to each index value and thus
    implies that a categorical approach should be taken

    """
    theme = colormap
    dim = len(dataframe)
    fig = make_subplots(rows=dim, cols=dim, print_grid=False)
    trace_list = []
    legend_param = 0
    for listy in dataframe:
        for listx in dataframe:
            unique_index_vals = {}
            for name in index_vals:
                if name not in unique_index_vals:
                    unique_index_vals[name] = []
            for name in sorted(unique_index_vals.keys()):
                new_listx = []
                new_listy = []
                for j in range(len(index_vals)):
                    if index_vals[j] == name:
                        new_listx.append(listx[j])
                        new_listy.append(listy[j])
                if legend_param == 1:
                    if listx == listy and diag == 'histogram':
                        trace = graph_objs.Histogram(x=new_listx, marker=dict(color=theme[name]), showlegend=True)
                    elif listx == listy and diag == 'box':
                        trace = graph_objs.Box(y=new_listx, name=None, marker=dict(color=theme[name]), showlegend=True)
                    elif 'marker' in kwargs:
                        kwargs['marker']['size'] = size
                        kwargs['marker']['color'] = theme[name]
                        trace = graph_objs.Scatter(x=new_listx, y=new_listy, mode='markers', name=name, showlegend=True, **kwargs)
                    else:
                        trace = graph_objs.Scatter(x=new_listx, y=new_listy, mode='markers', name=name, marker=dict(size=size, color=theme[name]), showlegend=True, **kwargs)
                elif listx == listy and diag == 'histogram':
                    trace = graph_objs.Histogram(x=new_listx, marker=dict(color=theme[name]), showlegend=False)
                elif listx == listy and diag == 'box':
                    trace = graph_objs.Box(y=new_listx, name=None, marker=dict(color=theme[name]), showlegend=False)
                elif 'marker' in kwargs:
                    kwargs['marker']['size'] = size
                    kwargs['marker']['color'] = theme[name]
                    trace = graph_objs.Scatter(x=new_listx, y=new_listy, mode='markers', name=name, showlegend=False, **kwargs)
                else:
                    trace = graph_objs.Scatter(x=new_listx, y=new_listy, mode='markers', name=name, marker=dict(size=size, color=theme[name]), showlegend=False, **kwargs)
                unique_index_vals[name] = trace
            trace_list.append(unique_index_vals)
            legend_param += 1
    trace_index = 0
    indices = range(1, dim + 1)
    for y_index in indices:
        for x_index in indices:
            for name in sorted(trace_list[trace_index].keys()):
                fig.append_trace(trace_list[trace_index][name], y_index, x_index)
            trace_index += 1
    for j in range(dim):
        xaxis_key = 'xaxis{}'.format(dim * dim - dim + 1 + j)
        fig['layout'][xaxis_key].update(title=headers[j])
    for j in range(dim):
        yaxis_key = 'yaxis{}'.format(1 + dim * j)
        fig['layout'][yaxis_key].update(title=headers[j])
    hide_tick_labels_from_box_subplots(fig)
    if diag == 'histogram':
        fig['layout'].update(height=height, width=width, title=title, showlegend=True, barmode='stack')
        return fig
    else:
        fig['layout'].update(height=height, width=width, title=title, showlegend=True)
        return fig