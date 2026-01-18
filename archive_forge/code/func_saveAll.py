import numpy as np
from ..Qt import QtCore, QtGui, QtWidgets
def saveAll(self):
    """Save all data to file."""
    self.save(self.serialize(useSelection=False))