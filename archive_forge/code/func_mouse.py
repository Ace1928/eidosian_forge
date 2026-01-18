import numpy as np
import verlet_chain
import pyqtgraph as pg
def mouse(pos):
    global mousepos
    pos = view.mapSceneToView(pos)
    mousepos = np.array([pos.x(), pos.y()])