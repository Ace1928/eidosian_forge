import warnings
import plotly.graph_objs as go
from plotly.matplotlylib.mplexporter import Renderer
from plotly.matplotlylib import mpltools
def draw_xlabel(self, **props):
    """Add an xaxis label to the current subplot in layout dictionary.

        props.keys() -- [
        'text',         (actual content string, not the text obj)
        'position',     (an x, y pair, not an mpl Bbox)
        'coordinates',  ('data', 'axes', 'figure', 'display')
        'text_type',    ('title', 'xlabel', or 'ylabel')
        'style',        (style dict, see below)
        'mplobj'        (actual mpl text object)
        ]

        props['style'].keys() -- [
        'alpha',        (opacity of text)
        'fontsize',     (size in points of text)
        'color',        (hex color)
        'halign',       (horizontal alignment, 'left', 'center', or 'right')
        'valign',       (vertical alignment, 'baseline', 'center', or 'top')
        'rotation',
        'zorder',       (precedence of text when stacked with other objs)
        ]

        """
    self.msg += '        Adding xlabel\n'
    axis_key = 'xaxis{0}'.format(self.axis_ct)
    self.plotly_fig['layout'][axis_key]['title'] = str(props['text'])
    titlefont = dict(size=props['style']['fontsize'], color=props['style']['color'])
    self.plotly_fig['layout'][axis_key]['titlefont'] = titlefont