import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.environ as pyo
from pyomo.contrib.viewer.model_browser import ComponentDataModel
import pyomo.contrib.viewer.qt as myqt
from pyomo.common.dependencies import DeferredImportError
def test_create_tree_expr(self):
    ui_data = UIData(model=self.m)
    data_model = ComponentDataModel(parent=None, ui_data=ui_data, components=(Expression,), columns=['name', 'value'])
    assert len(data_model.rootItems) == 1
    assert data_model.rootItems[0].data == self.m
    children = data_model.rootItems[0].children
    assert children[0].data == self.m.b1
    assert children[0].children[0].data == self.m.b1.e1
    ui_data.calculate_expressions()
    root_index = data_model.index(0, 0)
    b1_index = data_model.index(0, 0, parent=root_index)
    e1_index0 = data_model.index(0, 0, parent=b1_index)
    e1_index1 = data_model.index(0, 1, parent=b1_index)
    assert data_model.data(e1_index0) == 'b1.e1'
    assert abs(data_model.data(e1_index1) - 3.0) < 0.0001