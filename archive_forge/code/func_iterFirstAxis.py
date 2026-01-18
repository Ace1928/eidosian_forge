import numpy as np
from ..Qt import QtCore, QtGui, QtWidgets
def iterFirstAxis(self, data):
    for i in range(data.shape[0]):
        yield data[i]