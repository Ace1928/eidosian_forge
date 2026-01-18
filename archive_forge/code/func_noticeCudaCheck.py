import argparse
import itertools
import sys
import numpy as np
from utils import FrameCounter
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import VideoTemplate_generic as ui_template
def noticeCudaCheck():
    global xp, cache
    cache = {}
    if ui.cudaCheck.isChecked():
        if _has_cupy:
            xp = cp
        else:
            xp = np
            ui.cudaCheck.setChecked(False)
    else:
        xp = np
    mkData()