from pyomo.environ import (
import pyomo.common.unittest as unittest
import pyomo.contrib.viewer.qt as myqt
import pyomo.contrib.viewer.pyomo_viewer as pv
from pyomo.contrib.viewer.qt import available
@unittest.pytest.fixture(scope='module')
def qtbot():
    """Overwrite qtbot - remove test failure"""
    return