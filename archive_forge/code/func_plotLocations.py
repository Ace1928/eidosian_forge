from fontTools.varLib.models import VariationModel, supportScalar
from fontTools.designspaceLib import DesignSpaceDocument
from matplotlib import pyplot
from mpl_toolkits.mplot3d import axes3d
from itertools import cycle
import math
import logging
import sys
def plotLocations(locations, fig, names=None, **kwargs):
    n = len(locations)
    cols = math.ceil(n ** 0.5)
    rows = math.ceil(n / cols)
    if names is None:
        names = [None] * len(locations)
    model = VariationModel(locations)
    names = [names[model.reverseMapping[i]] for i in range(len(names))]
    axes = sorted(locations[0].keys())
    if len(axes) == 1:
        _plotLocations2D(model, axes[0], fig, cols, rows, names=names, **kwargs)
    elif len(axes) == 2:
        _plotLocations3D(model, axes, fig, cols, rows, names=names, **kwargs)
    else:
        raise ValueError('Only 1 or 2 axes are supported')