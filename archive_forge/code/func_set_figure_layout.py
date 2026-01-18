from collections import OrderedDict
from plotly import exceptions, optional_imports
from plotly.graph_objs import graph_objs
def set_figure_layout(self, width, height):
    """
        Sets and returns default layout object for dendrogram figure.

        """
    self.layout.update({'showlegend': False, 'autosize': False, 'hovermode': 'closest', 'width': width, 'height': height})
    self.set_axis_layout(self.xaxis)
    self.set_axis_layout(self.yaxis)
    return self.layout