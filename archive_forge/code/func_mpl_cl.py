from collections import OrderedDict
from itertools import chain
def mpl_cl(name, colorlist):
    cm[name] = ListedColormap(colorlist, name)
    register_cmap('cet_' + name, cmap=cm[name])
    return cm[name]