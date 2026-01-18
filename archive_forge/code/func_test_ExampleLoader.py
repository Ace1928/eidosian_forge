from ExampleApp import ExampleLoader
import pyqtgraph as pg
from pyqtgraph.Qt import QtTest
def test_ExampleLoader():
    loader = ExampleLoader()
    QtTest.QTest.qWaitForWindowExposed(loader)
    QtTest.QTest.qWait(200)
    loader.close()