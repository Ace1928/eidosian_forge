from collections import OrderedDict
from plotly import exceptions, optional_imports
from plotly.graph_objs import graph_objs
def get_color_dict(self, colorscale):
    """
        Returns colorscale used for dendrogram tree clusters.

        :param (list) colorscale: Colors to use for the plot in rgb format.
        :rtype (dict): A dict of default colors mapped to the user colorscale.

        """
    d = {'r': 'red', 'g': 'green', 'b': 'blue', 'c': 'cyan', 'm': 'magenta', 'y': 'yellow', 'k': 'black', 'w': 'white'}
    default_colors = OrderedDict(sorted(d.items(), key=lambda t: t[0]))
    if colorscale is None:
        rgb_colorscale = ['rgb(0,116,217)', 'rgb(35,205,205)', 'rgb(61,153,112)', 'rgb(40,35,35)', 'rgb(133,20,75)', 'rgb(255,65,54)', 'rgb(255,255,255)', 'rgb(255,220,0)']
    else:
        rgb_colorscale = colorscale
    for i in range(len(default_colors.keys())):
        k = list(default_colors.keys())[i]
        if i < len(rgb_colorscale):
            default_colors[k] = rgb_colorscale[i]
    new_old_color_map = [('C0', 'b'), ('C1', 'g'), ('C2', 'r'), ('C3', 'c'), ('C4', 'm'), ('C5', 'y'), ('C6', 'k'), ('C7', 'g'), ('C8', 'r'), ('C9', 'c')]
    for nc, oc in new_old_color_map:
        try:
            default_colors[nc] = default_colors[oc]
        except KeyError:
            default_colors[n] = 'rgb(0,116,217)'
    return default_colors