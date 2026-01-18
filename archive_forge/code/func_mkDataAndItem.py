import argparse
import itertools
import re
import numpy as np
from utils import FrameCounter
import pyqtgraph as pg
import pyqtgraph.parametertree as ptree
from pyqtgraph.Qt import QtCore, QtWidgets
@interactor.decorate(count=dict(limits=[1, None], step=100), size=dict(limits=[1, None]))
def mkDataAndItem(count=500, size=10):
    global data
    scale = 100
    data = {'pos': np.random.normal(size=(50, count), scale=scale), 'pen': [pg.mkPen(x) for x in np.random.randint(0, 256, (count, 3))], 'brush': [pg.mkBrush(x) for x in np.random.randint(0, 256, (count, 3))], 'size': (np.random.random(count) * size).astype(int)}
    data['pen'][0] = pg.mkPen('w')
    data['size'][0] = size
    data['brush'][0] = pg.mkBrush('b')
    bound = 5 * scale
    p.setRange(xRange=[-bound, bound], yRange=[-bound, bound])
    mkItem()