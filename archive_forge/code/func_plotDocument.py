from fontTools.varLib.models import VariationModel, supportScalar
from fontTools.designspaceLib import DesignSpaceDocument
from matplotlib import pyplot
from mpl_toolkits.mplot3d import axes3d
from itertools import cycle
import math
import logging
import sys
def plotDocument(doc, fig, **kwargs):
    doc.normalize()
    locations = [s.location for s in doc.sources]
    names = [s.name for s in doc.sources]
    plotLocations(locations, fig, names, **kwargs)