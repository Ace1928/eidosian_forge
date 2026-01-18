from collections import OrderedDict
from itertools import chain
def mpl_cm(name, colorlist):
    cm[name] = LinearSegmentedColormap.from_list(name, colorlist, N=len(colorlist))
    register_cmap('cet_' + name, cmap=cm[name])
    return cm[name]