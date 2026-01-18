import traceback
import numpy as np
import pyqtgraph as pg
from pyqtgraph.graphicsItems.ScatterPlotItem import name_list
from pyqtgraph.Qt import QtWidgets, QtCore
from pyqtgraph.parametertree import interact, ParameterTree, Parameter
import random
def isNoneOrScalar(value):
    return value is None or np.isscalar(value[0])