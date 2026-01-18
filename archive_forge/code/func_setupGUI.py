import collections
import os
import sys
from time import perf_counter
import numpy as np
import pyqtgraph as pg
from pyqtgraph import configfile
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree import types as pTypes
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
def setupGUI(self):
    self.layout = QtWidgets.QVBoxLayout()
    self.layout.setContentsMargins(0, 0, 0, 0)
    self.setLayout(self.layout)
    self.splitter = QtWidgets.QSplitter()
    self.splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
    self.layout.addWidget(self.splitter)
    self.tree = ParameterTree(showHeader=False)
    self.splitter.addWidget(self.tree)
    self.splitter2 = QtWidgets.QSplitter()
    self.splitter2.setOrientation(QtCore.Qt.Orientation.Vertical)
    self.splitter.addWidget(self.splitter2)
    self.worldlinePlots = pg.GraphicsLayoutWidget()
    self.splitter2.addWidget(self.worldlinePlots)
    self.animationPlots = pg.GraphicsLayoutWidget()
    self.splitter2.addWidget(self.animationPlots)
    self.splitter2.setSizes([int(self.height() * 0.8), int(self.height() * 0.2)])
    self.inertWorldlinePlot = self.worldlinePlots.addPlot()
    self.refWorldlinePlot = self.worldlinePlots.addPlot()
    self.inertAnimationPlot = self.animationPlots.addPlot()
    self.inertAnimationPlot.setAspectLocked(1)
    self.refAnimationPlot = self.animationPlots.addPlot()
    self.refAnimationPlot.setAspectLocked(1)
    self.inertAnimationPlot.setXLink(self.inertWorldlinePlot)
    self.refAnimationPlot.setXLink(self.refWorldlinePlot)