from pyomo.environ import (
import pyomo.common.unittest as unittest
import pyomo.contrib.viewer.qt as myqt
import pyomo.contrib.viewer.pyomo_viewer as pv
from pyomo.contrib.viewer.qt import available
@unittest.skipIf(not available, 'Qt packages are not available.')
def test_bad_view(qtbot):
    m = get_model()
    mw, m = get_mainwindow(model=m, testing=True)
    err = None
    try:
        mw.badTree = mw._tree_restart(w=mw.variables, standard='Bad Stuff', ui_data=mw.ui_data)
    except ValueError:
        err = 'ValueError'
    assert err == 'ValueError'