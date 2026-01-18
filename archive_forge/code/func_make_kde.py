from plotly import exceptions, optional_imports
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
def make_kde(self):
    """
        Makes the kernel density estimation(s) for create_distplot().

        This is called when curve_type = 'kde' in create_distplot().

        :rtype (list) curve: list of kde representations
        """
    curve = [None] * self.trace_number
    for index in range(self.trace_number):
        self.curve_x[index] = [self.start[index] + x * (self.end[index] - self.start[index]) / 500 for x in range(500)]
        self.curve_y[index] = scipy_stats.gaussian_kde(self.hist_data[index])(self.curve_x[index])
        if self.histnorm == ALTERNATIVE_HISTNORM:
            self.curve_y[index] *= self.bin_size[index]
    for index in range(self.trace_number):
        curve[index] = dict(type='scatter', x=self.curve_x[index], y=self.curve_y[index], xaxis='x1', yaxis='y1', mode='lines', name=self.group_labels[index], legendgroup=self.group_labels[index], showlegend=False if self.show_hist else True, marker=dict(color=self.colors[index % len(self.colors)]))
    return curve