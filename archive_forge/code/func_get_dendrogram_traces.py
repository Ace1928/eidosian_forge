from collections import OrderedDict
from plotly import exceptions, optional_imports
from plotly.graph_objs import graph_objs
def get_dendrogram_traces(self, X, colorscale, distfun, linkagefun, hovertext, color_threshold):
    """
        Calculates all the elements needed for plotting a dendrogram.

        :param (ndarray) X: Matrix of observations as array of arrays
        :param (list) colorscale: Color scale for dendrogram tree clusters
        :param (function) distfun: Function to compute the pairwise distance
                                   from the observations
        :param (function) linkagefun: Function to compute the linkage matrix
                                      from the pairwise distances
        :param (list) hovertext: List of hovertext for constituent traces of dendrogram
        :rtype (tuple): Contains all the traces in the following order:
            (a) trace_list: List of Plotly trace objects for dendrogram tree
            (b) icoord: All X points of the dendrogram tree as array of arrays
                with length 4
            (c) dcoord: All Y points of the dendrogram tree as array of arrays
                with length 4
            (d) ordered_labels: leaf labels in the order they are going to
                appear on the plot
            (e) P['leaves']: left-to-right traversal of the leaves

        """
    d = distfun(X)
    Z = linkagefun(d)
    P = sch.dendrogram(Z, orientation=self.orientation, labels=self.labels, no_plot=True, color_threshold=color_threshold)
    icoord = np.array(P['icoord'])
    dcoord = np.array(P['dcoord'])
    ordered_labels = np.array(P['ivl'])
    color_list = np.array(P['color_list'])
    colors = self.get_color_dict(colorscale)
    trace_list = []
    for i in range(len(icoord)):
        if self.orientation in ['top', 'bottom']:
            xs = icoord[i]
        else:
            xs = dcoord[i]
        if self.orientation in ['top', 'bottom']:
            ys = dcoord[i]
        else:
            ys = icoord[i]
        color_key = color_list[i]
        hovertext_label = None
        if hovertext:
            hovertext_label = hovertext[i]
        trace = dict(type='scatter', x=np.multiply(self.sign[self.xaxis], xs), y=np.multiply(self.sign[self.yaxis], ys), mode='lines', marker=dict(color=colors[color_key]), text=hovertext_label, hoverinfo='text')
        try:
            x_index = int(self.xaxis[-1])
        except ValueError:
            x_index = ''
        try:
            y_index = int(self.yaxis[-1])
        except ValueError:
            y_index = ''
        trace['xaxis'] = f'x{x_index}'
        trace['yaxis'] = f'y{y_index}'
        trace_list.append(trace)
    return (trace_list, icoord, dcoord, ordered_labels, P['leaves'])