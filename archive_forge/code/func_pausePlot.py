import argparse
import itertools
import re
import numpy as np
from utils import FrameCounter
import pyqtgraph as pg
import pyqtgraph.parametertree as ptree
from pyqtgraph.Qt import QtCore, QtWidgets
@interactor.decorate()
def pausePlot(paused=False):
    if paused:
        timer.stop()
    else:
        timer.start()