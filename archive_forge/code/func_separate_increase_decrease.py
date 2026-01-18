from plotly import exceptions
from plotly.graph_objs import graph_objs
from plotly.figure_factory import utils
def separate_increase_decrease(self):
    """
        Separate data into two groups: increase and decrease

        (1) Increase, where close > open and
        (2) Decrease, where close <= open
        """
    for index in range(len(self.open)):
        if self.close[index] is None:
            pass
        elif self.close[index] > self.open[index]:
            self.increase_x.append(self.all_x[index])
            self.increase_y.append(self.all_y[index])
        else:
            self.decrease_x.append(self.all_x[index])
            self.decrease_y.append(self.all_y[index])