from plotly.figure_factory import utils
from plotly.figure_factory._ohlc import (
from plotly.graph_objs import graph_objs
def get_candle_decrease(self):
    """
        Separate increasing data from decreasing data.

        The data is increasing when close value > open value
        and decreasing when the close value <= open value.
        """
    decrease_y = []
    decrease_x = []
    for index in range(len(self.open)):
        if self.close[index] <= self.open[index]:
            decrease_y.append(self.low[index])
            decrease_y.append(self.open[index])
            decrease_y.append(self.close[index])
            decrease_y.append(self.close[index])
            decrease_y.append(self.close[index])
            decrease_y.append(self.high[index])
            decrease_x.append(self.x[index])
    decrease_x = [[x, x, x, x, x, x] for x in decrease_x]
    decrease_x = utils.flatten(decrease_x)
    return (decrease_x, decrease_y)