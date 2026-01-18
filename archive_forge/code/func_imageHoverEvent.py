import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
def imageHoverEvent(event):
    """Show the position, pixel, and value under the mouse cursor.
    """
    if event.isExit():
        p1.setTitle('')
        return
    pos = event.pos()
    i, j = (pos.y(), pos.x())
    i = int(np.clip(i, 0, data.shape[0] - 1))
    j = int(np.clip(j, 0, data.shape[1] - 1))
    val = data[i, j]
    ppos = img.mapToParent(pos)
    x, y = (ppos.x(), ppos.y())
    p1.setTitle('pos: (%0.1f, %0.1f)  pixel: (%d, %d)  value: %.3g' % (x, y, i, j, val))