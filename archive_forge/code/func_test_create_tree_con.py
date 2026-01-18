import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.environ as pyo
from pyomo.contrib.viewer.model_browser import ComponentDataModel
import pyomo.contrib.viewer.qt as myqt
from pyomo.common.dependencies import DeferredImportError
def test_create_tree_con(self):
    ui_data = UIData(model=self.m)
    data_model = ComponentDataModel(parent=None, ui_data=ui_data, components=(Constraint,), columns=['name', 'active'])
    assert len(data_model.rootItems) == 1
    assert data_model.rootItems[0].data == self.m
    children = data_model.rootItems[0].children
    assert children[0].data == self.m.b1
    assert children[1].data == self.m.c1
    assert children[2].data == self.m.c2
    root_index = data_model.index(0, 0)
    assert data_model.data(root_index) == 'tm'
    idx = data_model.index(0, 0, parent=root_index)
    assert data_model.data(idx) == 'b1'
    idx = data_model.index(0, 1, parent=root_index)
    assert data_model.data(idx) == True
    idx = data_model.index(1, 0, parent=root_index)
    assert data_model.data(idx) == 'c1'
    idx = data_model.index(2, 0, parent=root_index)
    assert data_model.data(idx) == 'c2'
    c2 = idx.internalPointer()
    c2.set('active', False)
    assert not self.m.c2.active
    c2.set('active', True)
    assert self.m.c2.active
    assert 'z[2]' in data_model.data(idx, role=myqt.Qt.ItemDataRole.ToolTipRole)