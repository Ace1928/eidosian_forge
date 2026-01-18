from _buildParamTypes import makeAllParamTypes
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree
def bChanged(self):
    self.a.setValue(1.0 / self.b.value(), blockSignal=self.aChanged)