import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
def add_heading(lw, name):
    global num_bars
    lw.addLabel('=== ' + name + ' ===')
    num_bars += 1
    lw.nextRow()