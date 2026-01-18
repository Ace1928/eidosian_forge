import math
from plotly import exceptions, optional_imports
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
def blank_pos(self, xi, yi):
    """
        Set up positions for trajectories to be used with rk4 function.
        """
    return (int(xi / self.spacing_x + 0.5), int(yi / self.spacing_y + 0.5))