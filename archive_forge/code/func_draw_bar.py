import warnings
import plotly.graph_objs as go
from plotly.matplotlylib.mplexporter import Renderer
from plotly.matplotlylib import mpltools
def draw_bar(self, coll):
    """Draw a collection of similar patches as a bar chart.

        After bars are sorted, an appropriate data dictionary must be created
        to tell plotly about this data. Just like draw_line or draw_markers,
        draw_bar translates patch/path information into something plotly
        understands.

        Positional arguments:
        patch_coll -- a collection of patches to be drawn as a bar chart.

        """
    tol = 1e-10
    trace = [mpltools.make_bar(**bar_props) for bar_props in coll]
    widths = [bar_props['x1'] - bar_props['x0'] for bar_props in trace]
    heights = [bar_props['y1'] - bar_props['y0'] for bar_props in trace]
    vertical = abs(sum((widths[0] - widths[iii] for iii in range(len(widths))))) < tol
    horizontal = abs(sum((heights[0] - heights[iii] for iii in range(len(heights))))) < tol
    if vertical and horizontal:
        x_zeros = [bar_props['x0'] for bar_props in trace]
        if all((x_zeros[iii + 1] > x_zeros[iii] for iii in range(len(x_zeros[:-1])))):
            orientation = 'v'
        else:
            orientation = 'h'
    elif vertical:
        orientation = 'v'
    else:
        orientation = 'h'
    if orientation == 'v':
        self.msg += '    Attempting to draw a vertical bar chart\n'
        old_heights = [bar_props['y1'] for bar_props in trace]
        for bar in trace:
            bar['y0'], bar['y1'] = (0, bar['y1'] - bar['y0'])
        new_heights = [bar_props['y1'] for bar_props in trace]
        for old, new in zip(old_heights, new_heights):
            if abs(old - new) > tol:
                self.plotly_fig['layout']['barmode'] = 'stack'
                self.plotly_fig['layout']['hovermode'] = 'x'
        x = [bar['x0'] + (bar['x1'] - bar['x0']) / 2 for bar in trace]
        y = [bar['y1'] for bar in trace]
        bar_gap = mpltools.get_bar_gap([bar['x0'] for bar in trace], [bar['x1'] for bar in trace])
        if self.x_is_mpl_date:
            x = [bar['x0'] for bar in trace]
            formatter = self.current_mpl_ax.get_xaxis().get_major_formatter().__class__.__name__
            x = mpltools.mpl_dates_to_datestrings(x, formatter)
    else:
        self.msg += '    Attempting to draw a horizontal bar chart\n'
        old_rights = [bar_props['x1'] for bar_props in trace]
        for bar in trace:
            bar['x0'], bar['x1'] = (0, bar['x1'] - bar['x0'])
        new_rights = [bar_props['x1'] for bar_props in trace]
        for old, new in zip(old_rights, new_rights):
            if abs(old - new) > tol:
                self.plotly_fig['layout']['barmode'] = 'stack'
                self.plotly_fig['layout']['hovermode'] = 'y'
        x = [bar['x1'] for bar in trace]
        y = [bar['y0'] + (bar['y1'] - bar['y0']) / 2 for bar in trace]
        bar_gap = mpltools.get_bar_gap([bar['y0'] for bar in trace], [bar['y1'] for bar in trace])
    bar = go.Bar(orientation=orientation, x=x, y=y, xaxis='x{0}'.format(self.axis_ct), yaxis='y{0}'.format(self.axis_ct), opacity=trace[0]['alpha'], marker=go.bar.Marker(color=trace[0]['facecolor'], line=dict(width=trace[0]['edgewidth'])))
    if len(bar['x']) > 1:
        self.msg += '    Heck yeah, I drew that bar chart\n'
        (self.plotly_fig.add_trace(bar),)
        if bar_gap is not None:
            self.plotly_fig['layout']['bargap'] = bar_gap
    else:
        self.msg += '    Bar chart not drawn\n'
        warnings.warn('found box chart data with length <= 1, assuming data redundancy, not plotting.')