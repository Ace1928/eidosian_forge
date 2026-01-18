import argparse
import itertools
import re
import numpy as np
from utils import FrameCounter
import pyqtgraph as pg
import pyqtgraph.parametertree as ptree
from pyqtgraph.Qt import QtCore, QtWidgets
@interactor.decorate()
def mkItem(pxMode=True, useCache=True):
    global item
    item = pg.ScatterPlotItem(pxMode=pxMode, **getData())
    item.opts['useCache'] = useCache
    p.clear()
    p.addItem(item)