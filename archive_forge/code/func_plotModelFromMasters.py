from fontTools.varLib.models import VariationModel, supportScalar
from fontTools.designspaceLib import DesignSpaceDocument
from matplotlib import pyplot
from mpl_toolkits.mplot3d import axes3d
from itertools import cycle
import math
import logging
import sys
def plotModelFromMasters(model, masterValues, fig, **kwargs):
    """Plot a variation model and set of master values corresponding
    to the locations to the model into a pyplot figure.  Variation
    model must have axisOrder of size 1 or 2."""
    if len(model.axisOrder) == 1:
        _plotModelFromMasters2D(model, masterValues, fig, **kwargs)
    elif len(model.axisOrder) == 2:
        _plotModelFromMasters3D(model, masterValues, fig, **kwargs)
    else:
        raise ValueError('Only 1 or 2 axes are supported')