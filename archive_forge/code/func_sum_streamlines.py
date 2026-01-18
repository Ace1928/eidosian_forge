import math
from plotly import exceptions, optional_imports
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
def sum_streamlines(self):
    """
        Makes all streamlines readable as a single trace.

        :rtype (list, list): streamline_x: all x values for each streamline
            combined into single list and streamline_y: all y values for each
            streamline combined into single list
        """
    streamline_x = sum(self.st_x, [])
    streamline_y = sum(self.st_y, [])
    return (streamline_x, streamline_y)