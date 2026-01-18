from plotly.figure_factory import utils
from plotly.figure_factory._ohlc import (
from plotly.graph_objs import graph_objs
def get_candle_increase(self):
    """
        Separate increasing data from decreasing data.

        The data is increasing when close value > open value
        and decreasing when the close value <= open value.
        """
    increase_y = []
    increase_x = []
    for index in range(len(self.open)):
        if self.close[index] > self.open[index]:
            increase_y.append(self.low[index])
            increase_y.append(self.open[index])
            increase_y.append(self.close[index])
            increase_y.append(self.close[index])
            increase_y.append(self.close[index])
            increase_y.append(self.high[index])
            increase_x.append(self.x[index])
    increase_x = [[x, x, x, x, x, x] for x in increase_x]
    increase_x = utils.flatten(increase_x)
    return (increase_x, increase_y)