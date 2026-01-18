import math
from plotly import exceptions
from plotly.graph_objs import graph_objs
from plotly.figure_factory import utils
def get_barbs(self):
    """
        Creates x and y startpoint and endpoint pairs

        After finding the endpoint of each barb this zips startpoint and
        endpoint pairs to create 2 lists: x_values for barbs and y values
        for barbs

        :rtype: (list, list) barb_x, barb_y: list of startpoint and endpoint
            x_value pairs separated by a None to create the barb of the arrow,
            and list of startpoint and endpoint y_value pairs separated by a
            None to create the barb of the arrow.
        """
    self.end_x = [i + j for i, j in zip(self.x, self.u)]
    self.end_y = [i + j for i, j in zip(self.y, self.v)]
    empty = [None] * len(self.x)
    barb_x = utils.flatten(zip(self.x, self.end_x, empty))
    barb_y = utils.flatten(zip(self.y, self.end_y, empty))
    return (barb_x, barb_y)